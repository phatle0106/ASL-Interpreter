# I3D Inference

This project provides an implementation of I3D for video understanding.  
You can set up the environment and run inference with the provided script.

## Setup

Python version: **3.9.7**

### 1. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate       # on Windows


pip install -r requirements.txt
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2-f https://download.pytorch.org/whl/torch_stable.html
