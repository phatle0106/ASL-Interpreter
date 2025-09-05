import base64
import io
import os
import time
from threading import Lock
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pytorch_i3d import InceptionI3d
import torchvision.transforms as transforms
import videotransforms

# ===== Configuration (aligned with I3D/gem_infer.py) =====
CLIP_LEN = 64
NUM_CLASSES = 100
WEIGHTS_PATH = "checkpoint/slt_100_005624_0.756.pt"
MODE = 'rgb'
GLOSS_PATH = r'preprocess/wlasl_class_list.txt'

STRIDE = 5
VOTING_BAG_SIZE = 6
THRESHOLD = 0.605
BACKGROUND_CLASS_ID = -1


# ===== Utilities =====
def load_gloss_map(path: str) -> Dict[int, str]:
    gloss_map: Dict[int, str] = {}
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


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    # frame expected BGR uint8
    frame = cv2.resize(frame, (224, 224))
    frame = (frame / 255.0) * 2 - 1
    return frame


def frames_to_tensor(frames: List[np.ndarray], device: torch.device) -> torch.Tensor:
    frames_np = np.stack(frames, axis=0)                # (T,H,W,C)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))  # (C,T,H,W)
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)         # (1,C,T,H,W)
    return frames_tensor.to(device)


def load_model(device: torch.device) -> torch.nn.Module:
    model = InceptionI3d(400, in_channels=3)
    # Initial RGB weights
    model.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location=device))
    model.replace_logits(NUM_CLASSES)
    # Task-specific checkpoint
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = torch.nn.DataParallel(model)
    model.eval()
    return model


# ===== Session State =====
class SessionState:
    def __init__(self):
        self.frame_buffer: List[np.ndarray] = []
        self.raw_predictions_queue: List[int] = []
        self.last_confirmed_class_id: Optional[int] = None
        self.frame_counter: int = 0
        self.last_current_class_id: Optional[int] = None  # for current_gloss display


sessions: Dict[str, SessionState] = {}
sessions_lock = Lock()


# ===== FastAPI Schemas =====
class ProcessFrameRequest(BaseModel):
    session_id: str
    frame: str  # base64 image or data URL


class ResetSessionRequest(BaseModel):
    session_id: str


# ===== App Initialization =====
app = FastAPI(title="I3D ASL Microservice", version="0.1.0")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    gloss_map = load_gloss_map(GLOSS_PATH)
except Exception as e:
    gloss_map = {}
    print(f"Warning: failed to load gloss map at {GLOSS_PATH}: {e}")

model: Optional[torch.nn.Module] = None
model_loaded_error: Optional[str] = None

try:
    model = load_model(device)
except Exception as e:
    model_loaded_error = str(e)
    print(f"Failed to load I3D model: {e}")


def ensure_session(session_id: str) -> SessionState:
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = SessionState()
        return sessions[session_id]


def decode_base64_image(b64: str) -> np.ndarray:
    # Support data URLs (e.g., "data:image/jpeg;base64,...")
    if b64.startswith('data:'):
        try:
            header, b64 = b64.split(',', 1)
        except ValueError:
            pass
    try:
        img_bytes = base64.b64decode(b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('cv2.imdecode returned None')
        return img  # BGR
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok" if (model is not None and model_loaded_error is None) else "error",
        "model_loaded": model is not None and model_loaded_error is None,
        "device": str(device),
        "clip_len": CLIP_LEN,
        "stride": STRIDE,
        "threshold": THRESHOLD,
        "labels_version": os.path.basename(GLOSS_PATH),
        "error": model_loaded_error or False,
    }


@app.post("/reset_session")
def reset_session(req: ResetSessionRequest):
    with sessions_lock:
        if req.session_id in sessions:
            sessions.pop(req.session_id)
    return {"status": "reset", "session_id": req.session_id}


@app.post("/process_frame")
def process_frame(req: ProcessFrameRequest):
    if model is None or model_loaded_error is not None:
        raise HTTPException(status_code=503, detail=f"Model not available: {model_loaded_error}")
    if device.type != 'cuda':
        # Prefer GPU per deployment plan; allow CPU with warning
        print("Warning: running on CPU; CUDA not available")

    # Decode image
    try:
        frame_bgr = decode_base64_image(req.frame)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    state = ensure_session(req.session_id)

    # Preprocess and buffer frame
    frame_proc = preprocess_frame(frame_bgr)
    state.frame_buffer.append(frame_proc)
    if len(state.frame_buffer) > CLIP_LEN:
        state.frame_buffer.pop(0)

    state.frame_counter += 1

    recognition_text: Optional[str] = None
    current_gloss_text: str = ""

    # Only attempt inference at stride and with full buffer
    if len(state.frame_buffer) == CLIP_LEN and state.frame_counter % STRIDE == 0:
        with torch.no_grad():
            input_tensor = frames_to_tensor(state.frame_buffer, device)
            logits = model(input_tensor)
            predictions = torch.max(logits, dim=2)[0]
            probs = F.softmax(predictions, dim=1)
            max_prob, pred_class = torch.max(probs, dim=1)
            pred_class_id = int(pred_class.item())
            max_prob_val = float(max_prob.item())

            # Set current_gloss for UI regardless of threshold
            current_gloss_text = gloss_map.get(pred_class_id, f"Class_{pred_class_id}")
            state.last_current_class_id = pred_class_id

            if max_prob_val >= THRESHOLD:
                state.raw_predictions_queue.append(pred_class_id)
            else:
                state.raw_predictions_queue.append(BACKGROUND_CLASS_ID)

            if len(state.raw_predictions_queue) > VOTING_BAG_SIZE:
                state.raw_predictions_queue.pop(0)

        # Majority vote
        if len(state.raw_predictions_queue) == VOTING_BAG_SIZE:
            from collections import Counter
            vote_counts = Counter(state.raw_predictions_queue)
            majority_class_id, max_count = vote_counts.most_common(1)[0]
            if majority_class_id != BACKGROUND_CLASS_ID and max_count > VOTING_BAG_SIZE / 2:
                if majority_class_id != state.last_confirmed_class_id:
                    recognition_text = gloss_map.get(majority_class_id, f"Class_{majority_class_id}")
                    state.last_confirmed_class_id = majority_class_id
            else:
                state.last_confirmed_class_id = None

    # If no stride fire yet, optionally show last current class
    if not current_gloss_text and state.last_current_class_id is not None:
        current_gloss_text = gloss_map.get(state.last_current_class_id, f"Class_{state.last_current_class_id}")

    return {
        "session_id": req.session_id,
        "recognition": recognition_text,  # top-1 only when confirmed
        "current_gloss": current_gloss_text or "",
        "buffer_size": len(state.frame_buffer),
        "frame_counter": state.frame_counter,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "error": False,
    }


# Run with: uvicorn I3D.i3d_service:app --host 127.0.0.1 --port 5000

