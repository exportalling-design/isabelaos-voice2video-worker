FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# 🔥 VERSIONES COMPATIBLES ENTRE SÍ
RUN pip install mmengine==0.9.1

RUN pip install mmcv==2.1.0 -f \
https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

RUN pip install mmpose==1.2.0

# Verificación real
RUN python -c "import cv2, mmcv, mmengine, mmpose; print('OK_IMPORTS')"

WORKDIR /app
COPY worker.py /app/worker.py

CMD ["python", "/app/worker.py"]
