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
# Paths (en volumen)
# ---------------------------
# Monta tu network volume en /runpod-volume
MUSE_ROOT = os.environ.get("MUSE_ROOT", "/runpod-volume/MuseTalk")
VOICES_DIR = os.environ.get("VOICES_DIR", "/runpod-volume/voices")

FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV", f"{VOICES_DIR}/female_ref.wav")
MALE_REF_WAV   = os.environ.get("MALE_REF_WAV",   f"{VOICES_DIR}/male_ref.wav")

# (opcional) si MuseTalk requiere venv específico en volumen:
MUSE_PY = os.environ.get("MUSE_PY", "python")

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
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())

def _b64_to_file(b64: str, out_path: str):
    raw = _decode_b64(b64)
    with open(out_path, "wb") as f:
        f.write(raw)

def _run(cmd: list, cwd: str = None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        tail = p.stdout[-6000:] if p.stdout else ""
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout

# ---------------------------
# XTTS (Coqui TTS)
# ---------------------------
def _tts_xtts_to_wav(text: str, voice: str, lang: str, out_wav: str):
    # Lazy import
    from TTS.api import TTS
    import torch

    speaker_wav = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    if not os.path.isfile(speaker_wav):
        raise RuntimeError(f"Missing speaker_wav: {speaker_wav}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=lang,
        file_path=out_wav,
    )

# ---------------------------
# MuseTalk inference
# ---------------------------
def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    if not os.path.isdir(MUSE_ROOT):
        raise RuntimeError(f"MuseTalk root not found: {MUSE_ROOT}")

    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"])
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"])

    # MuseTalk command (ajusta si tu repo usa otro script/config)
    cmd = [
        MUSE_PY, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
        "--bbox_shift", "0",
        "--use_float16",
    ]
    _run(cmd, cwd=MUSE_ROOT)

    results_dir = os.path.join(MUSE_ROOT, "results", "v15")
    cand = os.path.join(results_dir, "input_face_audio.mp4")
    if os.path.isfile(cand):
        return cand

    # fallback: último mp4
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"No results dir: {results_dir}")

    mp4s = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".mp4")]
    if not mp4s:
        raise RuntimeError("MuseTalk no produjo mp4")
    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]

# ---------------------------
# Main generator
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

    # límites por duración (los mismos que vas a reflejar en UI)
    max_chars = int(inp.get("max_chars") or (45 if seconds == 3 else 80))
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

        _tts_xtts_to_wav(text=text, voice=voice, lang=lang, out_wav=tts_wav)
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
    }

# ---------------------------
# RunPod handler
# ---------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        ping = str(inp.get("ping") or inp.get("mode") or "").strip().lower()

        if ping in ("echo", "debug"):
            return {"ok": True, "msg": "ECHO_OK", "input": inp, "paths": {"MUSE_ROOT": MUSE_ROOT, "VOICES_DIR": VOICES_DIR}}

        if ping in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode/ping: {ping}"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
