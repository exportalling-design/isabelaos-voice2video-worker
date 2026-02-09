FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python3
RUN python3 -m pip install --upgrade pip

# Torch cu118 (para CUDA 11.8) dentro de la imagen
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Coqui TTS dentro de la imagen (para que no use el venv del volumen)
RUN pip install --no-cache-dir TTS

# Dependencias típicas para MuseTalk inference (mínimo razonable)
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
    omegaconf \
    transformers \
    accelerate \
    diffusers

# RunPod SDK
RUN pip install --no-cache-dir runpod

WORKDIR /app
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
