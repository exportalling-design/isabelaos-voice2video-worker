# -------- Dockerfile (RunPod Serverless) --------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Prefer /usr/local/bin/python3
RUN ln -sf /usr/bin/python3 /usr/local/bin/python3 && python3 -m pip install --upgrade pip

# --- Install Torch CU118 + TTS globally (system python) ---
# Esto evita el infierno de venv en serverless.
RUN pip install --no-cache-dir \
    torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Coqui TTS (XTTS)
RUN pip install --no-cache-dir TTS

# RunPod SDK
RUN pip install --no-cache-dir runpod

WORKDIR /app
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
