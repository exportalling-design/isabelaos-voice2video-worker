# Dockerfile
# Base real (existe) en Docker Hub: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false

# (Opcional pero recomendado) herramientas mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- SOLO lo que te falta: mmcv + mmengine + mmpose ----
# Usamos openmim para instalar el stack de OpenMMLab en binarios compatibles con torch/cuda del container.
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -U openmim
RUN mim install -y "mmengine>=0.10.0" "mmcv>=2.0.0" "mmpose>=1.3.0"

# Verificación rápida en build (si esto pasa, ya no vas a volver a ver ModuleNotFoundError)
RUN python -c "import cv2, mmcv, mmengine, mmpose; print('OK: cv2+mmcv+mmengine+mmpose')"

# Worker
WORKDIR /app
COPY worker.py /app/worker.py

CMD ["python", "/app/worker.py"]
