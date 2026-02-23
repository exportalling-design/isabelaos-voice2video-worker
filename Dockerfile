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

# ✅ RunPod SDK (SIN esto el worker muere al iniciar)
RUN pip install runpod

# OpenMMLab core
RUN pip install mmengine==0.10.4

# mmcv prebuilt cu121/torch2.1
RUN pip install mmcv==2.1.0 -f \
  https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# ✅ verificación en build (si esto pasa, el container al menos puede importar)
RUN python -c "import runpod; import cv2, mmcv, mmengine; print('OK_CONTAINER_IMPORTS')"

WORKDIR /app
COPY worker.py /app/worker.py
CMD ["python", "/app/worker.py"]
