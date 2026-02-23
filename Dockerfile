# Ajusta este FROM al mismo base que ya usas en tu endpoint actual.
# (No lo cambio a ciegas para no romper tu stack de WAN/FFmpeg/etc.)
FROM your-current-base-image:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV TOKENIZERS_PARALLELISM=false
ENV COQUI_TOS_AGREED=1

# (Opcional) deps OS comunes, por si a tu base le falta algo mínimo.
# Si ya lo tienes, puedes borrar este bloque.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# ✅ SOLO OpenMMLab core deps (NO reinstala MuseTalk, NO toca volumen)
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install -U openmim \
 && mim install mmengine \
 && mim install "mmcv>=2.0.1" \
 && mim install "mmpose>=1.1.0"

# Tu worker
COPY worker.py /app/worker.py
COPY tts_generate.py /app/tts_generate.py

CMD ["python3", "-u", "/app/worker.py"]
