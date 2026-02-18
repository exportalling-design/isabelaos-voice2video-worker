FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false
ENV COQUI_TOS_AGREED=1
ENV TTS_USE_GPU=1

# Cache bust
ARG CACHE_BUST=2026-02-16-02
RUN echo "CACHE_BUST=$CACHE_BUST"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git wget curl ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python3

# pip + setuptools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Torch CUDA 11.8
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Base deps (audio + cv2)
RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy \
    librosa \
    soundfile \
    tqdm \
    einops \
    pillow \
    pyyaml \
    omegaconf \
    opencv-python

# RunPod SDK
RUN python3 -m pip install --no-cache-dir runpod

# XTTS stack limpio
RUN python3 -m pip uninstall -y transformers tokenizers accelerate huggingface_hub sentencepiece TTS || true
RUN python3 -m pip install --no-cache-dir \
    transformers==4.36.2 \
    tokenizers==0.15.2 \
    accelerate==0.25.0 \
    huggingface_hub==0.20.3 \
    sentencepiece==0.1.99 \
    TTS==0.22.0

# ✅ MuseTalk deps (diffusers + mmpose stack)
RUN python3 -m pip install --no-cache-dir diffusers==0.25.1 safetensors==0.4.2
RUN python3 -m pip install --no-cache-dir openmim==0.3.9
RUN mim install -y "mmengine==0.10.3" "mmcv==2.1.0" "mmdet==3.2.0" "mmpose==1.3.2"

# sanity checks
RUN python3 -c "import pkg_resources; print('pkg_resources OK')"
RUN python3 -c "import librosa; print('librosa OK')"
RUN python3 -c "import cv2; print('cv2 OK')"
RUN python3 -c "import diffusers; print('diffusers OK')"
RUN python3 -c "import mmpose; print('mmpose OK')"
RUN python3 -c "import torch; print('torch OK')"

WORKDIR /app
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
