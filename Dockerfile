FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git libgl1 libglib2.0-0 \
    build-essential python3-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install "numpy<2" opencv-python-headless

RUN pip install runpod

# OpenMMLab core
RUN pip install mmengine==0.10.4

# mmcv prebuilt cu121/torch2.1
RUN pip install mmcv==2.1.0 -f \
  https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# MuseTalk lightweight deps (solo Python libs)
RUN pip install omegaconf==2.3.0 hydra-core==1.3.2 pyyaml==6.0.1 tqdm==4.66.1

# 🔥 SOLO ESTO AGREGAMOS
RUN pip install mmpose==1.3.2 --no-deps

# Verificación final REAL
RUN python -c "import cv2, mmcv, mmengine, mmpose; print('OK_FULL_CONTAINER')"

WORKDIR /app
COPY worker.py /app/worker.py
CMD ["python", "/app/worker.py"]
