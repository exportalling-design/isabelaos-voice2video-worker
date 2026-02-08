# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Solo lo mínimo que tu worker usa:
# - bash: porque tu worker hace ["bash","-lc", ...]
# - ffmpeg: por si MuseTalk o tu pipeline lo requiere en runtime
# - ca-certificates: para descargas https (video_url)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias livianas del worker (NO torch, NO diffusers)
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Tu código (liviano)
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python", "-u", "/app/worker.py"]
