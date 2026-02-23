FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg \
    libgl1 libglib2.0-0 \
    build-essential \
    python3-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -V
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -U "numpy<2" cython

# EXACTO lo que tienes en el pod
RUN pip install mmengine==0.10.4

RUN pip install mmcv==2.1.0 -f \
https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

RUN pip install mmpose

# Verificación real
RUN python -c "import torch, mmcv, mmengine, mmpose; print('OK_IMPORTS')"

WORKDIR /app
COPY worker.py /app/worker.py
CMD ["python", "/app/worker.py"]
