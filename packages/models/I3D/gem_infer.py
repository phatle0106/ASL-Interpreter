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
import requests
import json
from threading import Thread, Lock
import base64
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration
CLIP_LEN = 64
NUM_CLASSES = 100
WEIGHTS_PATH = "checkpoint/nslt_100_005624_0.756.pt"
MODE = 'rgb'
GLOSS_PATH = r'preprocess/wlasl_class_list.txt'

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
MIN_GLOSSES_FOR_GEMINI = int(os.getenv("MIN_GLOSSES_FOR_GEMINI", "5"))  # Minimum glosses before sending to Gemini
SEND_TIMEOUT = 10

STRIDE = 5
VOTING_BAG_SIZE = 6
THRESHOLD = 0.605
BACKGROUND_CLASS_ID = -1

# Session management
_session_states = {}
_session_buffers = {}
_buffers_lock = Lock()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class InferenceRequest(BaseModel):
    session_id: str
    image: str  # base64 encoded image

class InferenceResponse(BaseModel):
    recognition: Optional[str] = None
    current_gloss: Optional[str] = None
    confidence: Optional[float] = None
    sentence: Optional[str] = None
    glosses_count: int = 0
    glosses_buffer: List[str] = []
    ready_for_sentence: bool = False
    timestamp: str
    session_id: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gemini_configured: bool
    timestamp: str

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

@dataclass
class SessionState:
    frame_buffer: List = None
    raw_predictions_queue: List = None
    last_confirmed_class_id: Optional[int] = None
    confirmed_gloss_text: str = ""
    frame_counter: int = 0
    
    def __post_init__(self):
        if self.frame_buffer is None:
            self.frame_buffer = []
        if self.raw_predictions_queue is None:
            self.raw_predictions_queue = []

def get_or_create_session(session_id: str) -> SessionState:
    """Get or create session state for a given session ID"""
    if session_id not in _session_states:
        _session_states[session_id] = SessionState()
    return _session_states[session_id]

def get_or_create_glosses_buffer(session_id: str) -> List[str]:
    """Get or create glosses buffer for a session"""
    with _buffers_lock:
        if session_id not in _session_buffers:
            _session_buffers[session_id] = []
        return _session_buffers[session_id]

def reset_session(session_id: str):
    """Reset session state and buffer"""
    if session_id in _session_states:
        del _session_states[session_id]
    with _buffers_lock:
        if session_id in _session_buffers:
            del _session_buffers[session_id]

# =============================================================================
# GEMINI API FUNCTIONS
# =============================================================================

def _send_gemini_request(glosses_list: List[str], session_id: str) -> Optional[str]:
    """
    Send glosses to Gemini API and get a meaningful sentence
    """
    if not GEMINI_API_KEY:
        print(f"   ---> Error: GEMINI_API_KEY not found for session {session_id}!")
        return None
    
    try:
        # Create the prompt for Gemini
        glosses_text = " ".join(glosses_list)
        prompt = f"""You are a sign language interpreter. I will give you a sequence of sign language glosses (individual sign words), and you need to convert them into a natural, grammatically correct English sentence that conveys the intended meaning.

Glosses: {glosses_text}

Please provide a natural English sentence that represents what the person is trying to communicate through these signs. Focus on the meaning rather than literal word order, as sign language grammar differs from English grammar.

Respond with only the sentence, no additional explanation."""

        # Prepare the request payload for Gemini API
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 100,
            }
        }

        # Make the API request
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(GEMINI_API_URL, 
                               json=payload, 
                               headers=headers, 
                               timeout=SEND_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                sentence = result['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f"\n   🤖 GEMINI SENTENCE for {session_id}: {sentence}")
                print(f"   📝 From glosses ({len(glosses_list)}): {glosses_text}\n")
                return sentence
            else:
                print(f"   ---> Gemini: No candidates returned for session {session_id}")
        else:
            print(f"   ---> Gemini HTTP {response.status_code} for session {session_id}: {response.text}")
            
    except Exception as e:
        print(f"   ---> Error contacting Gemini for session {session_id}: {e}")
    
    return None

def add_gloss_and_check_gemini(gloss: str, session_id: str) -> Optional[str]:
    """
    Add a new gloss to collection and send to Gemini if we have enough
    Returns the sentence if generated, None otherwise.
    """
    with _buffers_lock:
        if session_id not in _session_buffers:
            _session_buffers[session_id] = []
        
        _session_buffers[session_id].append(gloss)
        current_count = len(_session_buffers[session_id])
        
        # If we have enough glosses, send to Gemini and clear buffer
        if current_count >= MIN_GLOSSES_FOR_GEMINI:
            glosses_to_send = list(_session_buffers[session_id])
            _session_buffers[session_id] = []  # Clear buffer after copying
            
            # Send to Gemini synchronously to return sentence immediately
            sentence = _send_gemini_request(glosses_to_send, session_id)
            return sentence
    
    return None

# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def load_gloss_map(path: str) -> Dict[int, str]:
    """Load gloss mapping from file"""
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

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess a single frame"""
    frame = cv2.resize(frame, (224, 224))
    frame = (frame / 255.0) * 2 - 1
    return frame

def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """Convert frames list to tensor"""
    frames_np = np.stack(frames, axis=0)                # (T,H,W,C)
    frames_np = np.transpose(frames_np, (3, 0, 1, 2))   # (C,T,H,W)
    frames_tensor = torch.from_numpy(frames_np).float()
    frames_tensor = transform(frames_tensor)
    frames_tensor = frames_tensor.unsqueeze(0)          # (1,C,T,H,W)
    return frames_tensor.cuda()

def load_model():
    """Load the I3D model"""
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

# =============================================================================
# GLOBAL VARIABLES AND INITIALIZATION
# =============================================================================

# Global model and gloss map
model = None
gloss_map = None
transform = transforms.Compose([videotransforms.CenterCrop(224)])

def initialize():
    """Initialize model and gloss map"""
    global model, gloss_map
    if model is None:
        model = load_model()
    if gloss_map is None:
        gloss_map = load_gloss_map(GLOSS_PATH)
    print("Initialization complete.")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(title="I3D Sign Language Recognition Service", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    initialize()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        gemini_configured=GEMINI_API_KEY is not None,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Main inference endpoint"""
    if model is None or gloss_map is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Get or create session state
        session_state = get_or_create_session(request.session_id)
        
        # Preprocess frame and add to buffer
        frame_proc = preprocess_frame(frame)
        session_state.frame_buffer.append(frame_proc)
        
        # Keep only CLIP_LEN frames
        if len(session_state.frame_buffer) > CLIP_LEN:
            session_state.frame_buffer.pop(0)
        
        session_state.frame_counter += 1
        
        # Get current glosses buffer
        current_glosses = get_or_create_glosses_buffer(request.session_id)
        
        # Initialize response
        response = InferenceResponse(
            session_id=request.session_id,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            glosses_count=len(current_glosses),
            glosses_buffer=list(current_glosses[-5:]) if len(current_glosses) > 0 else []  # Show last 5
        )
        
        # Only process if we have enough frames and it's time to process
        if len(session_state.frame_buffer) == CLIP_LEN and session_state.frame_counter % STRIDE == 0:
            with torch.no_grad():
                input_tensor = frames_to_tensor(session_state.frame_buffer)
                logits = model(input_tensor)
                predictions = torch.max(logits, dim=2)[0]
                probs = F.softmax(predictions, dim=1)
                max_prob, pred_class = torch.max(probs, dim=1)
                pred_class_id = pred_class.item()
                max_prob_val = max_prob.item()
                
                # Add confidence to response
                response.confidence = max_prob_val
                
                # Add prediction to queue based on threshold
                if max_prob_val >= THRESHOLD:
                    session_state.raw_predictions_queue.append(pred_class_id)
                else:
                    session_state.raw_predictions_queue.append(BACKGROUND_CLASS_ID)
                
                # Keep only VOTING_BAG_SIZE predictions
                if len(session_state.raw_predictions_queue) > VOTING_BAG_SIZE:
                    session_state.raw_predictions_queue.pop(0)
                
                # Voting mechanism
                if len(session_state.raw_predictions_queue) == VOTING_BAG_SIZE:
                    vote_counts = Counter(session_state.raw_predictions_queue)
                    majority_class_id, max_count = vote_counts.most_common(1)[0]
                    
                    if majority_class_id != BACKGROUND_CLASS_ID and max_count > VOTING_BAG_SIZE / 2:
                        if majority_class_id != session_state.last_confirmed_class_id:
                            # New gloss recognized
                            gloss = gloss_map.get(majority_class_id, f'Class_{majority_class_id}')
                            session_state.confirmed_gloss_text = gloss
                            session_state.last_confirmed_class_id = majority_class_id
                            
                            response.recognition = gloss
                            response.current_gloss = gloss
                            
                            print(f"   ✅ Recognized for {request.session_id}: {gloss}")
                            
                            # Try to get sentence from Gemini
                            sentence = add_gloss_and_check_gemini(gloss, request.session_id)
                            if sentence:
                                response.sentence = sentence
                                response.ready_for_sentence = True
                                print(f"   🤖 Generated sentence for {request.session_id}: {sentence}")
                    else:
                        # Background or no clear majority
                        session_state.confirmed_gloss_text = ""
                        session_state.last_confirmed_class_id = None
                        response.current_gloss = ""
        else:
            # Not enough frames yet or not time to process
            response.current_gloss = session_state.confirmed_gloss_text
        
        # Update response with current buffer state
        with _buffers_lock:
            if request.session_id in _session_buffers:
                current_glosses = _session_buffers[request.session_id]
                response.glosses_count = len(current_glosses)
                response.glosses_buffer = list(current_glosses[-5:]) if len(current_glosses) > 0 else []
        
        return response
        
    except Exception as e:
        print(f"Error in inference for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset/{session_id}")
async def reset_session_endpoint(session_id: str):
    """Reset a specific session"""
    reset_session(session_id)
    return {"status": "success", "message": f"Session {session_id} reset"}

@app.post("/reset")
async def reset_all_sessions():
    """Reset all sessions"""
    global _session_states, _session_buffers
    _session_states.clear()
    with _buffers_lock:
        _session_buffers.clear()
    return {"status": "success", "message": "All sessions reset"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = []
    for session_id, state in _session_states.items():
        with _buffers_lock:
            glosses_count = len(_session_buffers.get(session_id, []))
            current_glosses = _session_buffers.get(session_id, [])
        
        sessions.append({
            "session_id": session_id,
            "frames_in_buffer": len(state.frame_buffer),
            "glosses_count": glosses_count,
            "current_glosses": current_glosses[-5:] if current_glosses else [],
            "last_gloss": state.confirmed_gloss_text,
            "frame_counter": state.frame_counter
        })
    
    return {"sessions": sessions}

if __name__ == "__main__":
    print(f"Starting I3D microservice...")
    print(f"Gemini API configured: {GEMINI_API_KEY is not None}")
    print(f"Min glosses for Gemini: {MIN_GLOSSES_FOR_GEMINI}")
    print(f"Model weights path: {WEIGHTS_PATH}")
    print(f"Gloss map path: {GLOSS_PATH}")
    
    # Initialize before starting
    initialize()
    
    uvicorn.run(app, host="0.0.0.0", port=5000)