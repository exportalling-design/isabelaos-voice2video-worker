FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false
ENV COQUI_TOS_AGREED=1
ENV TTS_USE_GPU=1

ARG CACHE_BUST=2026-02-17-01
RUN echo "CACHE_BUST=$CACHE_BUST"

# ✅ Sistema + git + ffmpeg + libs para cv2
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl ca-certificates \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python3

# pip base
RUN python3 -m pip install --upgrade pip setuptools wheel

# ✅ Torch CUDA 11.8 (torch2.1.x)
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ✅ Dependencias base (incluye opencv)
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
    opencv-python \
    requests

# ✅ Diffusers (MuseTalk lo usa)
RUN python3 -m pip install --no-cache-dir diffusers==0.25.1 safetensors==0.4.2

# RunPod SDK
RUN python3 -m pip install --no-cache-dir runpod

# ✅ Stack XTTS (como lo tenías)
RUN python3 -m pip uninstall -y transformers tokenizers accelerate huggingface_hub sentencepiece TTS || true
RUN python3 -m pip install --no-cache-dir \
    transformers==4.36.2 \
    tokenizers==0.15.2 \
    accelerate==0.25.0 \
    huggingface_hub==0.20.3 \
    sentencepiece==0.1.99 \
    TTS==0.22.0

# ✅ OpenMMLab (mmpose) con wheels prebuilt (CU118 + Torch2.1)
RUN python3 -m pip install --no-cache-dir -U openmim==0.3.9

# mmengine + mmcv (prebuilt wheels)
RUN python3 -m pip install --no-cache-dir \
    mmengine==0.10.3

RUN python3 -m pip install --no-cache-dir \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# mmpose (+ mmdet recomendado)
RUN python3 -m pip install --no-cache-dir \
    mmdet==3.2.0 \
    mmpose==1.3.2

# ✅ sanity checks
RUN python3 -c "import torch; print('torch OK', torch.__version__)"
RUN python3 -c "import cv2; print('cv2 OK', cv2.__version__)"
RUN python3 -c "import diffusers; print('diffusers OK', diffusers.__version__)"
RUN python3 -c "import mmpose; print('mmpose OK')"

WORKDIR /app
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
