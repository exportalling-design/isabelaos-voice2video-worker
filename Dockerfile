FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

# ✅ Auto-acepta CPML (evita prompt y/n en serverless)
ENV COQUI_TOS_AGREED=1
ENV TTS_USE_GPU=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python3
RUN python3 -m pip install --upgrade pip

# Torch cu118 (CUDA 11.8)
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ---- deps mínimos para MuseTalk / utilidades (SIN transformers/diffusers aquí) ----
RUN pip install --no-cache-dir \
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
RUN pip install --no-cache-dir runpod

# ✅ LIMPIA cualquier transformers viejo/nuevo (por si quedó algo)
RUN pip uninstall -y transformers tokenizers accelerate || true

# ✅ STACK XTTS PINNEADO Y FORZADO AL FINAL (esto es lo importante)
# Estos pins evitan el "torch >= 2.4" y mantienen BeamSearchScorer disponible.
RUN pip install --no-cache-dir --force-reinstall \
    "transformers==4.36.2" \
    "tokenizers==0.15.2" \
    "accelerate==0.25.0" \
    "huggingface_hub==0.20.3" \
    "sentencepiece==0.1.99" \
    "TTS==0.22.0"

# ✅ sanity check en build (si esto falla, no se construye y no perdés tiempo)
RUN python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

WORKDIR /app
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
