# /app/worker.py
# RunPod Serverless Worker — IsabelaOS Voice2Video (XTTS + MuseTalk)
# - Usa 2 entornos distintos (2 python): xtts_env y musetalk_env
# - Busca automáticamente el mount real del Network Volume (a veces es /workspace, no /runpod-volume)
# - Soporta:
#    {"input":{"mode":"echo"}}  -> debug de paths
#    {"input":{"mode":"voice_to_video", "text":"...", "voice":"female|male", "lang":"es|en", "seconds":3|5, "video_b64":"data:..."}}
#    {"input":{"mode":"voice_to_video", "video_url":"https://..."}}

import os
import gc
import time
import base64
import binascii
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict

import runpod

# ---------------------------
# ENV hardening
# ---------------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------
# Auto-detect de volumen montado
# ---------------------------
def _detect_base() -> str:
    candidates = [
        os.environ.get("RUNPOD_VOLUME_PATH", "").strip(),
        os.environ.get("VOLUME_PATH", "").strip(),
        "/workspace",
        "/runpod-volume",
        "/mnt",
        "/data",
        "/volume",
        "/workspace/runpod-volume",
    ]

    # leer /proc/mounts para capturar mounts reales
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mp = parts[1]
                    if mp not in candidates:
                        candidates.append(mp)
    except Exception:
        pass

    # Heurística: si existen estas carpetas, es el mount correcto
    for base in [c for c in candidates if c]:
        if (
            os.path.isdir(os.path.join(base, "MuseTalk"))
            and os.path.isdir(os.path.join(base, "voices"))
        ):
            return base

    # fallback razonable
    return "/workspace"


BASE = _detect_base()

# ---------------------------
# Paths en volumen (con fallback)
# ---------------------------
MUSE_ROOT = os.environ.get("MUSE_ROOT", f"{BASE}/MuseTalk")
VOICES_DIR = os.environ.get("VOICES_DIR", f"{BASE}/voices")

FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV", f"{VOICES_DIR}/female_ref.wav")
MALE_REF_WAV = os.environ.get("MALE_REF_WAV", f"{VOICES_DIR}/male_ref.wav")

# ✅ Dos entornos (dos python)
TTS_PY = os.environ.get("TTS_PY", f"{BASE}/xtts_env/bin/python")
MUSE_PY = os.environ.get("MUSE_PY", f"{BASE}/musetalk_env/bin/python")

# ---------------------------
# Helpers
# ---------------------------
def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass

def _clamp_int(v, lo: int, hi: int, default: int) -> int:
    try:
        n = int(round(float(v)))
    except Exception:
        return default
    return max(lo, min(hi, n))

def _decode_b64(s: str) -> bytes:
    if not s:
        raise ValueError("b64 vacío")
    s = str(s).strip()
    # soporta data:video/mp4;base64,....
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()
    s = s.replace("-", "+").replace("_", "/")
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"b64 inválido: {e}")

def _download_to_file(url: str, out_path: str):
    # Ojo: en serverless algunas configs limitan DNS si el URL es raro.
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())

def _b64_to_file(b64: str, out_path: str):
    raw = _decode_b64(b64)
    with open(out_path, "wb") as f:
        f.write(raw)

def _run(cmd: list, cwd: str = None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if p.returncode != 0:
        tail = (p.stdout or "")[-8000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout or ""

# ---------------------------
# XTTS -> WAV (usando env XTTS)
# ---------------------------
def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV

    if not os.path.isfile(speaker):
        raise RuntimeError(f"Missing speaker_wav: {speaker}")

    if not os.path.isfile(TTS_PY):
        raise RuntimeError(f"TTS_PY not found: {TTS_PY}")

    # Ejecuta el script dentro del env XTTS
    # Asegurate que /app/tts_generate.py exista dentro del repo/image
    cmd = [
        TTS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav,
    ]
    _run(cmd)

# ---------------------------
# MuseTalk inference (usando env MuseTalk)
# ---------------------------
def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    if not os.path.isdir(MUSE_ROOT):
        raise RuntimeError(f"MuseTalk root not found: {MUSE_ROOT}")
    if not os.path.isfile(MUSE_PY):
        raise RuntimeError(f"MUSE_PY not found: {MUSE_PY}")

    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"])
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"])

    cmd = [
        MUSE_PY, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
        "--bbox_shift", "0",
        "--use_float16",
    ]
    _run(cmd, cwd=MUSE_ROOT)

    # MuseTalk suele dejar salida en results/v15/...
    results_dir = os.path.join(MUSE_ROOT, "results", "v15")

    # nombre típico
    cand = os.path.join(results_dir, "input_face_audio.mp4")
    if os.path.isfile(cand):
        return cand

    if not os.path.isdir(results_dir):
        raise RuntimeError(f"No results dir: {results_dir}")

    mp4s = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".mp4")]
    if not mp4s:
        raise RuntimeError("MuseTalk no produjo mp4")
    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]

# ---------------------------
# Main pipeline
# ---------------------------
def voice_to_video(inp: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    text = str(inp.get("text") or "").strip()
    if not text:
        raise RuntimeError("Falta text")

    voice = str(inp.get("voice") or "female").strip().lower()
    if voice not in ("female", "male"):
        voice = "female"

    lang = str(inp.get("lang") or "es").strip().lower()
    if lang not in ("es", "en"):
        lang = "es"

    seconds = _clamp_int(inp.get("seconds", 3), 3, 5, 3)
    seconds = 3 if seconds < 4 else 5

    max_chars = 45 if seconds == 3 else 80
    if len(text) > max_chars:
        text = text[:max_chars]

    video_b64 = inp.get("video_b64") or inp.get("video")
    video_url = str(inp.get("video_url") or "").strip()

    if not video_b64 and not video_url:
        raise RuntimeError("Falta video_b64 o video_url")

    with tempfile.TemporaryDirectory() as td:
        in_mp4 = os.path.join(td, "in.mp4")
        tts_wav = os.path.join(td, "tts.wav")

        if video_url:
            _download_to_file(video_url, in_mp4)
        else:
            _b64_to_file(str(video_b64), in_mp4)

        _tts_make_wav(text=text, voice=voice, lang=lang, out_wav=tts_wav)
        out_mp4_path = _musetalk_infer(input_mp4=in_mp4, audio_wav=tts_wav)

        with open(out_mp4_path, "rb") as f:
            mp4_bytes = f.read()

    return {
        "ok": True,
        "mode": "voice_to_video",
        "seconds": seconds,
        "voice": voice,
        "lang": lang,
        "text_len": len(text),
        "elapsed_s": round(time.time() - t0, 3),
        "video_b64": base64.b64encode(mp4_bytes).decode("utf-8"),
        "video_mime": "video/mp4",
        "base": BASE,
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or inp.get("ping") or "").strip().lower()

        if mode in ("echo", "debug"):
            return {
                "ok": True,
                "msg": "ECHO_OK",
                "base": BASE,
                "paths": {
                    "BASE": BASE,
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "TTS_PY": TTS_PY,
                    "MUSE_PY": MUSE_PY,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                },
                "checks": {
                    "base_exists": os.path.isdir(BASE),
                    "muse_root_exists": os.path.isdir(MUSE_ROOT),
                    "voices_dir_exists": os.path.isdir(VOICES_DIR),
                    "female_ref_exists": os.path.isfile(FEMALE_REF_WAV),
                    "male_ref_exists": os.path.isfile(MALE_REF_WAV),
                    "tts_py_exists": os.path.isfile(TTS_PY),
                    "muse_py_exists": os.path.isfile(MUSE_PY),
                },
            }

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo or mode=voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})