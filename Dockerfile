FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

# ✅ Auto-acepta CPML (evita prompt y/n en serverless)
ENV COQUI_TOS_AGREED=1
ENV TTS_USE_GPU=1

# ✅ Cache bust (cambialo cuando quieras forzar rebuild REAL)
ARG CACHE_BUST=2026-02-16-01
RUN echo "CACHE_BUST=$CACHE_BUST"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python3
RUN python3 -m pip install --no-cache-dir --upgrade pip

# ✅ FIX: pkg_resources (setuptools) requerido por librosa
RUN python3 -m pip install --no-cache-dir --upgrade setuptools wheel

# Torch cu118 (CUDA 11.8)
RUN python3 -m pip install --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Dependencias mínimas para MuseTalk + audio utils
RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python \
    scipy \
    librosa \
    soundfile \
    tqdm \
    einops \
    pillow \
    pyyaml \
    omegaconf

# RunPod SDK
RUN python3 -m pip install --no-cache-dir runpod

# ✅ LIMPIA y PINNEA stack XTTS (evita choques de transformers/tokenizers)
RUN python3 -m pip uninstall -y transformers tokenizers accelerate huggingface_hub sentencepiece TTS || true

RUN python3 -m pip install --no-cache-dir --force-reinstall \
    "transformers==4.36.2" \
    "tokenizers==0.15.2" \
    "accelerate==0.25.0" \
    "huggingface_hub==0.20.3" \
    "sentencepiece==0.1.99" \
    "TTS==0.22.0"

# ✅ Pruebas DURAS en build (si falla aquí, no perdés tiempo con endpoint)
RUN python3 -c "import pkg_resources; print('pkg_resources OK')"
RUN python3 -c "import librosa; print('librosa OK')"
RUN python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

WORKDIR /app
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
