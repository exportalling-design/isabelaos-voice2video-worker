FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false
ENV COQUI_TOS_AGREED=1
ENV TTS_USE_GPU=1

# Cache bust (cambia el valor cuando quieras forzar rebuild)
ARG CACHE_BUST=2026-02-17-02
RUN echo "CACHE_BUST=$CACHE_BUST"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git \
    wget curl ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python3

# pip toolchain
RUN python3 -m pip install --upgrade pip setuptools wheel

# Torch CUDA 11.8 (torch 2.1)
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Base deps (audio + utils)
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

# XTTS stack limpio y compatible
RUN python3 -m pip uninstall -y transformers tokenizers accelerate huggingface_hub sentencepiece TTS || true
RUN python3 -m pip install --no-cache-dir \
    transformers==4.36.2 \
    tokenizers==0.15.2 \
    accelerate==0.25.0 \
    huggingface_hub==0.20.3 \
    sentencepiece==0.1.99 \
    TTS==0.22.0

# MuseTalk deps que te estaban faltando
RUN python3 -m pip install --no-cache-dir \
    diffusers==0.27.2 \
    safetensors

# OpenMMLab stack (mmpose requiere mmcv/mmengine)
RUN python3 -m pip install --no-cache-dir -U openmim

RUN mim install "mmengine==0.7.4"

RUN mim install "mmcv==2.0.1" \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

RUN mim install "mmdet==3.1.0"
RUN mim install "mmpose==1.1.0"

# Pruebas mínimas (si falla aquí, te enteras en build)
RUN python3 -c "import pkg_resources; print('pkg_resources OK')"
RUN python3 -c "import librosa; print('librosa OK')"
RUN python3 -c "import torch; print('torch OK', torch.__version__)"
RUN python3 -c "import diffusers; print('diffusers OK', diffusers.__version__)"
RUN python3 -c "import mmpose; print('mmpose OK')"
RUN python3 -c "from TTS.api import TTS; print('TTS import OK')"

WORKDIR /app
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
