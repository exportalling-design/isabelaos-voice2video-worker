FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false
ENV COQUI_TOS_AGREED=1
ENV TTS_USE_GPU=1

# Cache bust
ARG CACHE_BUST=2026-02-17-01
RUN echo "CACHE_BUST=$CACHE_BUST"

# System deps (incluye git + toolchain por si pip necesita compilar algo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl ca-certificates \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python3

# pip stack
RUN python3 -m pip install --upgrade pip setuptools wheel

# ✅ TORCH (MATCH para OpenMMLab wheels cu118/torch2.1)
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Base deps
RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 scipy librosa soundfile tqdm einops pillow pyyaml omegaconf \
    opencv-python

# RunPod SDK
RUN python3 -m pip install --no-cache-dir runpod

# XTTS stack (tu stack estable)
RUN python3 -m pip uninstall -y transformers tokenizers accelerate huggingface_hub sentencepiece TTS || true
RUN python3 -m pip install --no-cache-dir \
    transformers==4.36.2 \
    tokenizers==0.15.2 \
    accelerate==0.25.0 \
    huggingface_hub==0.20.3 \
    sentencepiece==0.1.99 \
    TTS==0.22.0

# ✅ OpenMMLab (la forma correcta en serverless)
RUN python3 -m pip install --no-cache-dir openmim==0.3.9

# ✅ mmcv wheel EXACTO (cu118 + torch2.1)
RUN python3 -m pip install --no-cache-dir \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# ✅ mmengine + mmdet + mmpose
RUN python3 -m pip install --no-cache-dir \
    mmengine==0.10.3 \
    mmdet==3.2.0 \
    mmpose==1.3.2

# Sanity checks (para que truene en build si falta algo)
RUN python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
RUN python3 -c "import cv2; print('cv2 OK')"
RUN python3 -c "import mmpose, mmdet, mmcv, mmengine; print('OpenMMLab OK')"
RUN python3 -c "from TTS.api import TTS; print('TTS OK')"

WORKDIR /app
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
