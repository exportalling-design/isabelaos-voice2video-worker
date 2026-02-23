FROM runpod/pytorch:2.1.0-cuda12.1-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install -U openmim \
 && mim install mmengine \
 && mim install "mmcv>=2.0.1" \
 && mim install "mmpose>=1.1.0"

COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
