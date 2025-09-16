### Bright of Sign — Real‑time ASL Word Recognition to Text

Bright of Sign is a real‑time American Sign Language (ASL) word‑level recognition project built on the I3D architecture. It recognizes glosses (word labels) from webcam video and can optionally turn the recognized gloss sequence into a natural English sentence using Google's Gemini Flash API.

There are two runnable apps:
- **Local Inference (`local_infer.py`)**: Minimal OpenCV window; shows live recognized gloss. Optional background segmentation via MediaPipe.
- **Streamlit + Gemini (`gem_infer.py`)**: Streamlit UI that collects glosses and sends them to Gemini Flash to generate a complete sentence.


## Repo layout
- `models/I3D/` — I3D model code, inference scripts, configs, and assets
  - `local_infer.py` — OpenCV realtime demo (minimal UI)
  - `gem_infer.py` — Streamlit UI + Gemini Flash sentence generation
  - `pytorch_i3d.py` — I3D model definition
  - `videotransforms.py` — basic transforms
  - `preprocess/` — class list and JSONs; uses `wlasl_class_list.txt`
  - `weights/` — pretrained I3D weights (e.g., `rgb_imagenet.pt`)
  - `checkpoint/` — fine‑tuned checkpoints (e.g., `nslt_100_005624_0.756.pt`)
  - `requirements.txt` — base dependencies for inference


## Quick demo (optional)
- Put a short screen recording or GIF of each flow here:
  - Local OpenCV demo → `docs/assets/local_infer_demo.gif`
  - Streamlit UI flow → `docs/assets/streamlit_ui.gif`

Once you add them, reference like:

```markdown
![Local demo](docs/assets/local_infer_demo.gif)
![Streamlit UI](docs/assets/streamlit_ui.gif)
```


## Requirements
- OS: Windows, macOS, or Linux
- Python: 3.9.x recommended (tested with 3.9.7)
- GPU optional (CUDA) — improves FPS
- Webcam


## Setup
1) Clone
```bash
git clone <this-repo-url>
cd nhom7-bright-of-sign
```

2) Create and activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3) Install Python dependencies
```bash
cd models/I3D
pip install -r requirements.txt

# Torch: pick a build for your platform/CUDA (examples)
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# or CUDA 11.x example (adjust to your GPU/CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Extras used by the apps (if not present from requirements)
pip install streamlit mediapipe
```

Notes:
- `models/I3D/requirements.txt` lists core libs (OpenCV, numpy, scikit‑learn, requests, Pillow, Google API clients, FastAPI/uvicorn), but you may still need to explicitly install a matching Torch build and Streamlit/MediaPipe as shown above.


## Model files
The inference scripts expect these files relative to `models/I3D/`:
- Pretrained I3D weights: `weights/rgb_imagenet.pt`
- Fine‑tuned checkpoint: `checkpoint/nslt_100_005624_0.756.pt` (default path in both apps)
- Class list (gloss map): `preprocess/wlasl_class_list.txt`

These files are included in this repo. If you change paths, update the constants in the scripts:
- In `local_infer.py`: `WEIGHTS_PATH`, `GLOSS_PATH`
- In `gem_infer.py`: `WEIGHTS_PATH`, `GLOSS_PATH`


## Environment variables (for Gemini)
To use the Streamlit + Gemini app, create a `.env` file in `models/I3D/` with:
```dotenv
GEMINI_API_KEY=your_api_key_here
```
Get an API key from Google AI Studio (Gemini).


## How to run

### 1) Local OpenCV app (minimal UI)
Runs a realtime window with the recognized gloss overlay.
```bash
cd models/I3D
python local_infer.py
```
- Quit: press `q` in the OpenCV window
- Tunables (edit in `local_infer.py`):
  - `CLIP_LEN` (default 64 frames), `STRIDE`, `VOTING_BAG_SIZE`, `THRESHOLD`
  - `USE_VIRTUAL_BG`, `BG_PATH`, `USE_MEDIAPIPE`

### 2) Streamlit app + Gemini sentence generation
Launch the UI, start/stop capture, generate sentences from collected glosses.
```bash
cd models/I3D
streamlit run gem_infer.py
```
- Make sure `.env` contains `GEMINI_API_KEY`
- UI controls:
  - Start / Stop: toggles webcam capture and buffering
  - Generate Sentence: sends collected glosses to Gemini Flash
  - Reset All: clears buffers, messages, and sentences


## How it works (high level)
- I3D backbone runs on sliding windows of `CLIP_LEN` frames, outputs class logits.
- Softmax + majority voting over a small queue stabilizes predictions.
- `local_infer.py` overlays the confirmed gloss on the video.
- `gem_infer.py` collects glosses and calls Gemini Flash to produce a natural English sentence.


## Tips for better results
- Ensure good lighting and clear framing of hands and upper body.
- Keep hands within the camera view; avoid fast motion blur.
- A GPU significantly improves throughput; otherwise reduce `CLIP_LEN` or increase `STRIDE` if needed.


## Troubleshooting
- Torch install issues: choose a wheel matching your Python/CUDA as shown above.
- Camera cannot open: ensure only one app is using the webcam and the correct index `cv2.VideoCapture(0)`.
- Low FPS on CPU: reduce `CLIP_LEN`, increase `STRIDE`, or run with a GPU.
- Gemini errors in Streamlit: verify `.env` and network access; check console for HTTP status.
- Missing files: check `weights/`, `checkpoint/`, and `preprocess/wlasl_class_list.txt` paths.
 - Streamlit × MediaPipe protobuf conflict: If you install both `streamlit` and `mediapipe` in the same env, you may hit protobuf version conflicts. Workarounds:
   - Recommended: use separate virtual environments
     - Env A (local demo with MediaPipe): install `mediapipe` and run `local_infer.py`.
     - Env B (Streamlit + Gemini): install `streamlit` and run `gem_infer.py`.
   - Or pin compatible versions (example; adjust per your platform):
     ```bash
     pip install "protobuf==3.20.3" "mediapipe==0.10.*" "streamlit==1.32.*"
     ```
     If Streamlit requires a newer protobuf, relax MediaPipe or upgrade both incrementally until they resolve. Prefer separate envs if you don’t need MediaPipe in Streamlit.


## Contributing / Next steps
- Add more classes or train on larger datasets (`archived/` contains configs/weights for other class counts).
- Improve the UI/UX, add background removal, and export recognized text.
- Add `docs/assets/*.gif` demo media for quick onboarding.

## Contributors
- Le Tran Tan Phat [(@phatle0106)](https://github.com/phatle0106)
- Nguyen Gia Huy (@)
- Nguyen Ho Quang Khai (@)
- Chu Minh Nguyen (@)
- Ho Hong Phuc Nguyen (@)


## Acknowledgements
Built as part of Python & ML 2025 Course, EE - ML & IoT Laboratory, Ho Chi Minh city University of Technology

## License
Add your license information here.
