#!/usr/bin/env bash
set -e

echo "[BOOT] IsabelaOS voice2video worker starting..."

# --------- detectar volumen ----------
VOL="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
if [ ! -d "$VOL" ]; then
  VOL="/runpod-volume"
fi

# --------- carpetas ----------
PIPER_DIR="$VOL/voices/piper/es"
mkdir -p "$PIPER_DIR"

FEMALE_ONNX="$PIPER_DIR/female.onnx"
FEMALE_JSON="$PIPER_DIR/female.onnx.json"
MALE_ONNX="$PIPER_DIR/male.onnx"
MALE_JSON="$PIPER_DIR/male.onnx.json"

# URLs directas (HuggingFace)
FEMALE_URL_ONNX="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/medium/es_ES-medium.onnx"
FEMALE_URL_JSON="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/medium/es_ES-medium.onnx.json"

MALE_URL_ONNX="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/medium/es_ES-mls_10246-medium.onnx"
MALE_URL_JSON="https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/medium/es_ES-mls_10246-medium.onnx.json"

download_if_missing () {
  local out="$1"
  local url="$2"
  local name="$3"

  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "[OK] $name exists: $out"
    return 0
  fi

  echo "[DL] Descargando $name..."
  curl -L --retry 5 --retry-delay 2 -o "$out" "$url"

  if [ ! -s "$out" ]; then
    echo "[FATAL] Descarga falló para $name"
    exit 1
  fi

  echo "[OK] $name listo"
}

# --------- descarga modelos ----------
download_if_missing "$FEMALE_ONNX" "$FEMALE_URL_ONNX" "female.onnx"
download_if_missing "$FEMALE_JSON" "$FEMALE_URL_JSON" "female.onnx.json"
download_if_missing "$MALE_ONNX"   "$MALE_URL_ONNX"   "male.onnx"
download_if_missing "$MALE_JSON"   "$MALE_URL_JSON"   "male.onnx.json"

# --------- exports ----------
export VOICES_DIR="$VOL/voices"
export PIPER_FEMALE_MODEL="$FEMALE_ONNX"
export PIPER_MALE_MODEL="$MALE_ONNX"
export PIPER_FEMALE_JSON="$FEMALE_JSON"
export PIPER_MALE_JSON="$MALE_JSON"

echo "[BOOT] Piper listo"
echo "  female -> $PIPER_FEMALE_MODEL"
echo "  male   -> $PIPER_MALE_MODEL"

# --------- arrancar worker ----------
python3 -u /app/worker.py
