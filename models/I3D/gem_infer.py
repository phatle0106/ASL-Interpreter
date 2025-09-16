# -*- coding: utf-8 -*-

import streamlit as st
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
import requests
import json
from PIL import Image
import base64
from io import BytesIO

load_dotenv()

# Model parameters
CLIP_LEN = 64
NUM_CLASSES = 100
WEIGHTS_PATH = "checkpoint/nslt_100_005624_0.756.pt"
MODE = 'rgb'
GLOSS_PATH = r'preprocess/wlasl_class_list.txt'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API configuration
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
SEND_TIMEOUT = 10

# Recognition parameters
STRIDE = 5
VOTING_BAG_SIZE = 6
THRESHOLD = 0.61
BACKGROUND_CLASS_ID = -1

st.set_page_config(
    page_title="ASL Recognition with AI Sentence Generation",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---------- Dark palette ---------- */
    :root {
        --bg-0: #0b1220;           /* app background */
        --bg-1: #0f172a;           /* panels */
        --bg-2: #111827;           /* inset panels */
        --bg-3: #0b1220;           /* video bg */
        --card: #0f172a;
        --muted: #94a3b8;          /* slate-400 */
        --text: #e5e7eb;           /* gray-200 */
        --text-weak: #cbd5e1;      /* slate-300 */
        --border: #1f2937;         /* gray-800 */
        --border-strong: #334155;  /* slate-700 */
        --shadow: rgba(0,0,0,.45);

        /* brand accents */
        --acc-green-1: #22c55e;
        --acc-green-2: #16a34a;
        --acc-blue-1: #3b82f6;
        --acc-blue-2: #2563eb;
        --acc-red-1: #ef4444;
        --acc-red-2: #dc2626;
        --acc-amber-1: #f59e0b;
        --acc-amber-2: #d97706;
        --acc-purple-1: #a855f7;
        --acc-purple-2: #7c3aed;

        --ok-bg: rgba(34,197,94,.12);
        --ok-border: #16a34a;
        --ok-text: #86efac;

        --sys-bg: rgba(59,130,246,.12);
        --sys-border: #3b82f6;
        --sys-text: #93c5fd;

        --err-bg: rgba(239,68,68,.12);
        --err-border: #ef4444;
        --err-text: #fecaca;

        --scrollbar: #334155;
        --scrollbar-track: #0b1220;
    }

    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .stApp {
        background: radial-gradient(1200px 800px at 20% -10%, #111827 0%, var(--bg-0) 40%, #0a0f1a 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--text);
    }

    /* -------- Video shell -------- */
    .video-container {
        background: var(--bg-3);
        border: 2px solid var(--border);
        border-radius: 24px;
        padding: 0;
        overflow: hidden;
        box-shadow: 0 10px 30px var(--shadow);
    }

    .video-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        border-bottom: 2px solid var(--border);
        background: linear-gradient(180deg, #0f172a, #0b1220);
    }

    .video-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--acc-red-1);
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(0,0,0,.25) inset;
    }
    .status-dot.connected {
        background: var(--acc-green-1);
        box-shadow: 0 0 10px rgba(34,197,94,.35);
    }

    .badge {
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 999px;
        border: 2px solid var(--border-strong);
        color: var(--text-weak);
        background: #0b1220;
        margin: 0 4px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    .badge.ok {
        color: var(--ok-text);
        border-color: var(--ok-border);
        background: var(--ok-bg);
    }

    /* -------- Panels -------- */
    .control-panel {
        background: var(--card);
        border: 2px solid var(--border);
        border-radius: 24px;
        padding: 16px;
        box-shadow: 0 10px 25px var(--shadow);
        color: var(--text);
    }

    .info-panel {
        background: var(--bg-2);
        border: 2px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        color: var(--text);
    }

    .info-title {
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.05em;
    }

    .info-content {
        font-size: 16px;
        color: var(--text);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        min-height: 24px;
        word-break: break-word;
        line-height: 1.5;
    }

    .sentence-display {
        background: linear-gradient(135deg, rgba(34,197,94,.14), rgba(16,185,129,.10));
        border: 2px solid var(--ok-border);
    }
    .sentence-display .info-content {
        font-weight: 600;
        font-size: 18px;
        color: #a7f3d0; /* emerald-200 */
    }

    /* -------- Messages -------- */
    .messages-container {
        background: var(--bg-2);
        border: 2px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        max-height: 400px;
        overflow-y: auto;
        margin-top: 16px;
    }
    .messages-container::-webkit-scrollbar {
        width: 10px; height: 10px;
    }
    .messages-container::-webkit-scrollbar-thumb {
        background: var(--scrollbar); border-radius: 10px;
    }
    .messages-container::-webkit-scrollbar-track {
        background: var(--scrollbar-track);
    }

    .message {
        margin: 8px 0;
        padding: 12px 16px;
        border-radius: 12px;
        border: 2px solid var(--border);
        max-width: 85%;
        font-family: 'Inter', sans-serif;
        color: var(--text);
        background: #0b1220;
    }

    .message.gloss {
        background: rgba(168,85,247,.12);
        border-color: var(--acc-purple-1);
        margin-left: auto;
        text-align: right;
        color: #d8b4fe; /* purple-300 */
        font-weight: 500;
    }

    .message.sentence {
        background: rgba(34,197,94,.12);
        border-color: var(--acc-green-1);
        margin: 0 auto;
        font-weight: 600;
        color: #bbf7d0; /* green-200 */
        text-align: center;
    }

    .message.system {
        background: rgba(59,130,246,.12);
        border-color: var(--acc-blue-1);
        margin: 0 auto;
        color: #bfdbfe; /* blue-200 */
        font-size: 14px;
        text-align: center;
        font-weight: 500;
    }

    .stats-overlay {
        position: absolute;
        top: 8px;
        left: 8px;
        background: rgba(2,6,23,0.9); /* slate-950 */
        color: var(--text);
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border: 1px solid var(--border);
        backdrop-filter: blur(2px);
    }

    /* -------- Buttons -------- */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, var(--acc-green-1), var(--acc-green-2));
        color: white;
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 10px;
        padding: 10px 16px;
        font-weight: 600;
        transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
        width: 100%;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        letter-spacing: 0.025em;
        min-height: 44px;
        box-shadow: 0 6px 16px rgba(16,185,129,.25);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(16,185,129,.35);
        filter: brightness(1.02);
    }

    .primary-btn button {
        background: linear-gradient(135deg, var(--acc-green-1), var(--acc-green-2)) !important;
    }
    .secondary-btn button {
        background: linear-gradient(135deg, var(--acc-amber-1), var(--acc-amber-2)) !important;
        box-shadow: 0 6px 16px rgba(245,158,11,.20) !important;
    }
    .secondary-btn button:hover {
        box-shadow: 0 10px 22px rgba(245,158,11,.30) !important;
    }
    .danger-btn button {
        background: linear-gradient(135deg, var(--acc-red-1), var(--acc-red-2)) !important;
        box-shadow: 0 6px 16px rgba(239,68,68,.20) !important;
    }
    .danger-btn button:hover {
        box-shadow: 0 10px 22px rgba(239,68,68,.30) !important;
    }

    /* -------- Streamlit overrides -------- */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li {
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stSuccess {
        background-color: var(--ok-bg) !important;
        border: 1px solid var(--ok-border) !important;
        color: var(--ok-text) !important;
    }
    .stInfo {
        background-color: var(--sys-bg) !important;
        border: 1px solid var(--sys-border) !important;
        color: var(--sys-text) !important;
    }
    .stError {
        background-color: var(--err-bg) !important;
        border: 1px solid var(--err-border) !important;
        color: var(--err-text) !important;
    }

    .stExpander {
        background-color: var(--card) !important;
        border: 2px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
    }
    .stExpander > div > div > div {
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'glosses_buffer' not in st.session_state:
    st.session_state.glosses_buffer = []

if 'model' not in st.session_state:
    st.session_state.model = None

if 'gloss_map' not in st.session_state:
    st.session_state.gloss_map = {}

if 'frame_buffer' not in st.session_state:
    st.session_state.frame_buffer = []

if 'raw_predictions_queue' not in st.session_state:
    st.session_state.raw_predictions_queue = []

if 'last_confirmed_class_id' not in st.session_state:
    st.session_state.last_confirmed_class_id = None

if 'confirmed_gloss_text' not in st.session_state:
    st.session_state.confirmed_gloss_text = ""

if 'frame_counter' not in st.session_state:
    st.session_state.frame_counter = 0

if 'generated_sentences' not in st.session_state:
    st.session_state.generated_sentences = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

if 'ai_connected' not in st.session_state:
    st.session_state.ai_connected = False

if 'current_sentence' not in st.session_state:
    st.session_state.current_sentence = "Ready to generate..."


@st.cache_resource
def load_gloss_map(path):
    """Load gloss mapping from file"""
    gloss_map = {}
    try:
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
    except FileNotFoundError:
        st.error(f"Gloss file not found: {path}")
        st.stop()
    return gloss_map

@st.cache_resource
def load_model():
    """Load the I3D model"""
    try:
        with st.spinner("Loading I3D model..."):
            model = InceptionI3d(400, in_channels=3)
            model.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location='cpu'))
            model.replace_logits(NUM_CLASSES)
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
            if torch.cuda.is_available():
                model.cuda()
            model = torch.nn.DataParallel(model)
            model.eval()
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    frame = cv2.resize(frame, (224, 224))
    frame = (frame / 255.0) * 2 - 1
    return frame

def frames_to_tensor(frames):
    """Convert frames list to tensor"""
    transform = transforms.Compose([videotransforms.CenterCrop(224)])
    frames_np = np.stack(frames, axis=0)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)
    
    if torch.cuda.is_available():
        return frames_tensor.cuda()
    return frames_tensor

def send_gemini_request(glosses_list):
    """Send glosses to Gemini API and get a meaningful sentence"""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not found!"
    
    try:
        glosses_text = " ".join(glosses_list)
        prompt = f"""You are a sign language interpreter. I will give you a sequence of sign language glosses (individual sign words), and you need to convert them into a natural, grammatically correct English sentence that conveys the intended meaning.

Glosses: {glosses_text}

Please provide a natural English sentence that represents what the person is trying to communicate through these signs. Focus on the meaning rather than literal word order, as sign language grammar differs from English grammar.

Respond with only the sentence, no additional explanation."""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 100,
            }
        }

        headers = {"Content-Type": "application/json"}
        
        response = requests.post(GEMINI_API_URL, 
                               json=payload, 
                               headers=headers, 
                               timeout=SEND_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                sentence = result['candidates'][0]['content']['parts'][0]['text'].strip()
                return sentence
            else:
                return "Gemini: No response generated"
        else:
            return f"Gemini HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def add_message(text, msg_type='system'):
    """Add message to session state"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.messages.append({
        'text': text,
        'type': msg_type,
        'timestamp': timestamp
    })

def process_frame_for_recognition(frame):
    """Process frame and return recognition result"""
    if st.session_state.model is None:
        return None
        
    frame_proc = preprocess_frame(frame)
    st.session_state.frame_buffer.append(frame_proc)
    
    if len(st.session_state.frame_buffer) > CLIP_LEN:
        st.session_state.frame_buffer.pop(0)
    
    st.session_state.frame_counter += 1
    
    if len(st.session_state.frame_buffer) == CLIP_LEN and st.session_state.frame_counter % STRIDE == 0:
        with torch.no_grad():
            input_tensor = frames_to_tensor(st.session_state.frame_buffer)
            logits = st.session_state.model(input_tensor)
            predictions = torch.max(logits, dim=2)[0]
            probs = F.softmax(predictions, dim=1)
            max_prob, pred_class = torch.max(probs, dim=1)
            pred_class_id = pred_class.item()
            max_prob_val = max_prob.item()

            if max_prob_val >= THRESHOLD:
                st.session_state.raw_predictions_queue.append(pred_class_id)
            else:
                st.session_state.raw_predictions_queue.append(BACKGROUND_CLASS_ID)

            if len(st.session_state.raw_predictions_queue) > VOTING_BAG_SIZE:
                st.session_state.raw_predictions_queue.pop(0)

        if len(st.session_state.raw_predictions_queue) == VOTING_BAG_SIZE:
            vote_counts = Counter(st.session_state.raw_predictions_queue)
            majority_class_id, max_count = vote_counts.most_common(1)[0]
            if majority_class_id != BACKGROUND_CLASS_ID and max_count > VOTING_BAG_SIZE / 2:
                if majority_class_id != st.session_state.last_confirmed_class_id:
                    gloss = st.session_state.gloss_map.get(majority_class_id, f'Class_{majority_class_id}')
                    st.session_state.confirmed_gloss_text = gloss
                    st.session_state.last_confirmed_class_id = majority_class_id
                    
                    st.session_state.glosses_buffer.append(gloss)
                    add_message(f"Recognized: {gloss}", 'gloss')
                    return gloss
            else:
                st.session_state.confirmed_gloss_text = ""
                st.session_state.last_confirmed_class_id = None
    
    return None
# if "yolo_seg" not in st.session_state:
#     st.session_state.yolo_seg = YOLO("yolov8m-seg.pt")

# Hàm segment người + thay background
def segment_person(frame, bg_color=(0, 0, 139)):
    # results = st.session_state.yolo_seg(frame, verbose=False)
    # output = np.full_like(frame, bg_color, dtype=np.uint8)

    # for r in results:
    #     if r.masks is not None:
    #         for mask, cls in zip(r.masks.data, r.boxes.cls):
    #             if int(cls) == 0:  # "person"
    #                 mask = mask.cpu().numpy()
    #                 mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    #                 mask = (mask > 0.4).astype(np.uint8)
    #                 for c in range(3):
    #                     output[:, :, c] = frame[:, :, c] * mask + output[:, :, c] * (1 - mask)

    return frame


def main():
    # --- Global CSS (font setup) ---
    st.markdown("""
    <style>
        * {
            font-family: "Segoe UI", "Roboto", "Helvetica", sans-serif;
        }
        .info-title {
            font-weight: 600;
            font-size: 16px;
        }
        .info-content {
            font-size: 14px;
        }
        .messages-container {
            font-size: 13px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="video-header" style="margin-bottom: 20px;">
        <div class="video-title">
            <span class="status-dot connected"></span>
            <span>ASL Recognition with AI Sentence Generation</span>
        </div>
        <div>
            <span class="badge ok">I3D Ready</span>
            <span class="badge ok">Gemini Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    fps_time = time.time()
    fps_count = 0
    fps = 0

    # Check API key
    if not GEMINI_API_KEY:
        st.error("⚠️ GEMINI_API_KEY not found! Please create a .env file with your API key.")
        st.markdown("Get your API key at: https://makersuite.google.com/app/apikey")
        return

    # Load model + gloss map
    if st.session_state.model is None:
        st.session_state.model = load_model()
        st.session_state.gloss_map = load_gloss_map(GLOSS_PATH)
        if st.session_state.model is not None:
            add_message("AI models loaded successfully!", 'system')

    # Init session state
    if "capture_active" not in st.session_state:
        st.session_state.capture_active = False
    if "frame_counter" not in st.session_state:
        st.session_state.frame_counter = 0
    if "frame_buffer" not in st.session_state:
        st.session_state.frame_buffer = []
    if "raw_predictions_queue" not in st.session_state:
        st.session_state.raw_predictions_queue = []
    if "last_confirmed_class_id" not in st.session_state:
        st.session_state.last_confirmed_class_id = None
    if "confirmed_gloss_text" not in st.session_state:
        st.session_state.confirmed_gloss_text = ""

    # Layout
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)

        # --- Toggle Start/Stop button ---
        if st.button("▶️ Start / ⏹ Stop"):
            if st.session_state.capture_active:
                # Stop
                st.session_state.capture_active = False
                st.session_state.frame_buffer = []
                st.session_state.raw_predictions_queue = []
                st.session_state.last_confirmed_class_id = None
                st.success("🛑 Capture stopped, buffers reset.")
            else:
                # Start
                st.session_state.capture_active = True
                st.session_state.frame_buffer = []
                st.session_state.raw_predictions_queue = []
                st.session_state.glosses_buffer = []
                st.session_state.last_confirmed_class_id = None
                st.session_state.frame_counter = 0
                st.success("▶️ Capture started.")

        # --- Camera loop ---
        frame_placeholder = st.empty()
        if st.session_state.capture_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Không mở được camera. Vui lòng kiểm tra thiết bị.")
                return

            while st.session_state.capture_active:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Không đọc được frame từ camera.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # --- Segment + replace background ---
                frame_processed = segment_person(frame_rgb)
                fps_count += 1
                if time.time() - fps_time >= 1.0:
                    fps = fps_count
                    fps_count = 0
                    fps_time = time.time()
                # Recognition
                recognized_gloss = process_frame_for_recognition(frame_processed)
                display_frame = frame_processed.copy()

                # Hiển thị video
                cv2.putText(display_frame, f'FPS: {fps}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame_placeholder.image(display_frame, channels="RGB")

                # Hiển thị gloss nếu có
                if recognized_gloss:
                    st.success(f"Recognized gloss: {recognized_gloss}")

            cap.release()

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-panel">
            <div class="info-title">System Status</div>
            <div class="info-content">
                Capture: {"Active" if st.session_state.capture_active else "Stopped"} | 
                Glosses Collected: {len(st.session_state.glosses_buffer)} | 
                Sentences Generated: {len(st.session_state.generated_sentences) if 'generated_sentences' in st.session_state else 0}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Control buttons ---
    spacer1, spacer2, col_btn1, col_btn2 = st.columns([1.1, 0.9, 1, 1])

    with col_btn1:
        if st.button("🤖 Generate Sentence", disabled=len(st.session_state.glosses_buffer) == 0):
            if st.session_state.glosses_buffer:
                with st.spinner("Generating sentence..."):
                    sentence = send_gemini_request(st.session_state.glosses_buffer)
                    st.session_state.current_sentence = sentence
                    if "generated_sentences" not in st.session_state:
                        st.session_state.generated_sentences = []
                    st.session_state.generated_sentences.append({
                        "glosses": list(st.session_state.glosses_buffer),
                        "sentence": sentence,
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                    add_message(sentence, 'sentence')
                st.success("✅ Sentence generated!")

    with col_btn2:
        if st.button("🔄 Reset All"):
            st.session_state.glosses_buffer = []
            st.session_state.messages = []
            st.session_state.generated_sentences = []
            st.session_state.current_sentence = "Ready to generate..."
            st.session_state.frame_buffer = []
            st.session_state.raw_predictions_queue = []
            st.session_state.last_confirmed_class_id = None
            st.session_state.capture_active = False
            st.success("🔄 Session reset!")

    # --- Right panel ---
    with col2:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-panel">
            <div class="info-title">Current Glosses ({len(st.session_state.glosses_buffer)})</div>
            <div class="info-content">{' '.join(st.session_state.glosses_buffer) if st.session_state.glosses_buffer else 'No glosses collected yet'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-panel sentence-display">
            <div class="info-title">Latest Generated Sentence</div>
            <div class="info-content">{st.session_state.current_sentence if 'current_sentence' in st.session_state else 'Ready to generate...'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="messages-container">', unsafe_allow_html=True)
        st.markdown("**Activity Log:**")

        for msg in st.session_state.messages[-10:]:
            msg_class = f"message {msg['type']}"
            if msg['type'] == 'gloss':
                content = f"**Gloss:** {msg['text']}"
            elif msg['type'] == 'sentence':
                content = f"**Generated:** \"{msg['text']}\""
            else:
                content = msg['text']

            st.markdown(f"""
            <div class="{msg_class}">
                <small>{msg['timestamp']}</small><br>
                {content}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Instructions ---
    with st.expander("📋 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. Bấm "Start / Stop" để bật camera và bắt đầu capture
        2. Khi capture đang chạy → hệ thống sẽ gom glosses bằng sliding window
        3. Bấm "Start / Stop" lần nữa để dừng capture
        4. Nhấn "Generate Sentence" để gửi glosses sang Gemini và sinh câu
        5. Dùng "Reset All" để xóa toàn bộ dữ liệu cũ

        **Tips:**
        - Cần đủ 64 frames để model I3D predict được gloss
        - Hệ thống chỉ hiển thị gloss khi đã được confirm qua voting bag
        - Reset All nếu muốn bắt đầu lại từ đầu
        """)

if __name__ == '__main__':
    main()