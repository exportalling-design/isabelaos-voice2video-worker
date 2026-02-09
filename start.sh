#!/usr/bin/env bash
set -e

echo "[BOOT] IsabelaOS voice2video worker starting..."

# --------- detectar volumen ----------
VOL="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
if [ ! -d "$VOL" ]; then
  VOL="/runpod-volume"
fi

# --------- carpetas ----------
PIPER_DIR="$VOL/voices/piper"
mkdir -p "$PIPER_DIR"

FEMALE_MODEL="$PIPER_DIR/female.onnx"
MALE_MODEL="$PIPER_DIR/male.onnx"

# 🔴 PEGA AQUÍ TUS URLs DIRECTAS A LOS .onnx
FEMALE_URL="PEGAR_URL_DIRECTO_FEMALE_ONNX"
MALE_URL="PEGAR_URL_DIRECTO_MALE_ONNX"

download_if_missing () {
  local out="$1"
  local url="$2"
  local name="$3"

  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "[OK] $name exists: $out"
    return 0
  fi

  if [[ "$url" == PEGAR_* ]] || [ -z "$url" ]; then
    echo "[FATAL] Falta URL para $name"
    exit 1
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
download_if_missing "$FEMALE_MODEL" "$FEMALE_URL" "female.onnx"
download_if_missing "$MALE_MODEL"   "$MALE_URL"   "male.onnx"

# --------- exports ----------
export VOICES_DIR="$VOL/voices"
export PIPER_FEMALE_MODEL="$FEMALE_MODEL"
export PIPER_MALE_MODEL="$MALE_MODEL"

echo "[BOOT] Piper listo"
echo "  female -> $PIPER_FEMALE_MODEL"
echo "  male   -> $PIPER_MALE_MODEL"

# --------- arrancar worker ----------
python3 -u /app/worker.py
