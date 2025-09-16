# -*- coding: utf-8 -*-

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_i3d import InceptionI3d
import torchvision.transforms as transforms
import videotransforms
import time
from collections import Counter
import os
from dotenv import load_dotenv

# Thêm Mediapipe
import mediapipe as mp

# =============================================================================
# CẢNH BÁO BẢO MẬT QUAN TRỌNG
# =============================================================================
load_dotenv()

# ======================= CẤU HÌNH & THAM SỐ =======================
CLIP_LEN = 64
NUM_CLASSES = 100
WEIGHTS_PATH = "checkpoint/nslt_100_005624_0.756.pt"
MODE = 'rgb'
GLOSS_PATH = r'preprocess/wlasl_class_list.txt'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

STRIDE = 4
VOTING_BAG_SIZE = 6
THRESHOLD = 0.61
BACKGROUND_CLASS_ID = -1

# Cấu hình nền ảo
USE_VIRTUAL_BG = True
BG_PATH = 'background.jpg'
USE_MEDIAPIPE = True  # <-- Bật/tắt Mediapipe
dark_colors = {
    "dark_gray":   (50, 50, 50),
    "dark_blue":   (100, 50, 0),
    "dark_green":  (50, 100, 50),
    "dark_red":    (50, 50, 150),
    "dark_purple": (100, 50, 100),
    "dark_cyan":   (100, 100, 50),
    "dark_brown":  (50, 80, 120),
}
FALLBACK_BG_COLOR = dark_colors["dark_gray"]

# ======================= CÁC HÀM TIỆN ÍCH =======================
def load_gloss_map(path):
    gloss_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            class_id = int(parts[0])
            gloss = ' '.join(parts[1:])
            gloss_map[class_id] = gloss
    return gloss_map

transform = transforms.Compose([videotransforms.CenterCrop(224)])

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = (frame / 255.0) * 2 - 1
    return frame

def load_model():
    print("Loading I3D model...")
    model = InceptionI3d(400, in_channels=3)
    model.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
    model.replace_logits(NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()
    print("Model loaded successfully.")
    return model

def frames_to_tensor(frames):
    frames_np = np.stack(frames, axis=0)                # (T,H,W,C)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))   # (C,T,H,W)
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)          # (1,C,T,H,W)
    return frames_tensor.cuda()

# ======================= VÒNG LẶP CHÍNH =======================
def main():
    if not GEMINI_API_KEY:
        print("CẢNH BÁO: Không tìm thấy GEMINI_API_KEY. Vui lòng tạo file .env và làm theo hướng dẫn ở đầu code.")
        return

    gloss_map = load_gloss_map(GLOSS_PATH)
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam.")
        return

    # Mediapipe chỉ khởi tạo nếu USE_MEDIAPIPE = True
    if USE_MEDIAPIPE:
        mp_selfie = mp.solutions.selfie_segmentation
        selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)

    # Chuẩn bị ảnh nền
    bg_image = None
    if USE_VIRTUAL_BG and os.path.exists(BG_PATH):
        bg_image = cv2.imread(BG_PATH)
    if bg_image is None:
        bg_resized_static = None
    else:
        bg_resized_static = bg_image

    frame_buffer = []
    raw_predictions_queue = []
    last_confirmed_class_id = None
    confirmed_gloss_text = ""
    frame_counter = 0

    fps_time = time.time()
    fps_count = 0
    fps = 0

    print("\nStarting real-time recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # resize background fallback
        if bg_resized_static is not None:
            bg_resized = cv2.resize(bg_resized_static, (W, H))
        else:
            bg_resized = np.full((H, W, 3), FALLBACK_BG_COLOR, dtype=np.uint8)

        # ======== Inference (I3D) ========
        frame_proc = preprocess_frame(frame)
        frame_buffer.append(frame_proc)
        if len(frame_buffer) > CLIP_LEN:
            frame_buffer.pop(0)

        frame_counter += 1
        if len(frame_buffer) == CLIP_LEN and frame_counter % STRIDE == 0:
            with torch.no_grad():
                input_tensor = frames_to_tensor(frame_buffer)
                logits = model(input_tensor)
                predictions = torch.max(logits, dim=2)[0]
                probs = F.softmax(predictions, dim=1)
                max_prob, pred_class = torch.max(probs, dim=1)
                pred_class_id = pred_class.item()
                max_prob_val = max_prob.item()

                if max_prob_val >= THRESHOLD:
                    raw_predictions_queue.append(pred_class_id)
                else:
                    raw_predictions_queue.append(BACKGROUND_CLASS_ID)

                if len(raw_predictions_queue) > VOTING_BAG_SIZE:
                    raw_predictions_queue.pop(0)

            if len(raw_predictions_queue) == VOTING_BAG_SIZE:
                vote_counts = Counter(raw_predictions_queue)
                majority_class_id, max_count = vote_counts.most_common(1)[0]
                if majority_class_id != BACKGROUND_CLASS_ID and max_count > VOTING_BAG_SIZE / 2:
                    if majority_class_id != last_confirmed_class_id:
                        gloss = gloss_map.get(majority_class_id, f'Class_{majority_class_id}')
                        confirmed_gloss_text = gloss
                        last_confirmed_class_id = majority_class_id
                        print(f"   ---> Recognized: {gloss}")
                else:
                    confirmed_gloss_text = ""
                    last_confirmed_class_id = None

        # ======== Mediapipe background replace (nếu bật) ========
        if USE_MEDIAPIPE:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = selfie_seg.process(rgb_frame)
            mask = results.segmentation_mask  # float32 [0..1]

            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            mask = (mask > 0.40).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            mask_3c = mask[..., None]
            display_frame = np.where(mask_3c == 1, frame, bg_resized)
        else:
            display_frame = frame.copy()  # Nếu không dùng Mediapipe, hiển thị frame gốc

        # ======== Overlay text ========
        if len(frame_buffer) < CLIP_LEN:
            cv2.putText(display_frame, "Collecting frames...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif confirmed_gloss_text:
            cv2.putText(display_frame, confirmed_gloss_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        fps_count += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_time = time.time()
        cv2.putText(display_frame, f'FPS: {fps}', (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Real-time Sign Language Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if USE_MEDIAPIPE:
        selfie_seg.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == '__main__':
    main()
