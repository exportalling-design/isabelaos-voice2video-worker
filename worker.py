# worker.py — IsabelaOS Voice2Video (XTTS + MuseTalk) — RunPod Serverless
# ✅ Serverless mounts Network Volume at: /workspace (NOT /runpod-volume)
# Modes:
#  - {"input":{"mode":"debug"}}  -> returns paths
#  - {"input":{"mode":"voice_to_video", ...}} -> returns mp4 as base64

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

def _detect_base():
    # candidatos típicos
    candidates = [
        os.environ.get("RUNPOD_VOLUME_PATH", "").strip(),
        "/workspace",
        "/runpod-volume",
        "/mnt",
        "/data",
        "/volume",
        "/workspace/runpod-volume",
    ]
    # también inspecciona mounts reales
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

    # elige el que tenga tu estructura
    for base in [c for c in candidates if c]:
        if os.path.isdir(os.path.join(base, "MuseTalk")) and os.path.isdir(os.path.join(base, "voices")):
            return base

    # fallback
    return "/workspace"

BASE = _detect_base()

# ---------------------------
# ENV hardening
# ---------------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------
# Paths (Serverless => /workspace)
# ---------------------------
MUSE_ROOT = os.environ.get("MUSE_ROOT", "/workspace/MuseTalk")
VOICES_DIR = os.environ.get("VOICES_DIR", "/workspace/voices")

FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV", f"{VOICES_DIR}/female_ref.wav")
MALE_REF_WAV = os.environ.get("MALE_REF_WAV", f"{VOICES_DIR}/male_ref.wav")

# ✅ Two separate venvs
TTS_PY = os.environ.get("TTS_PY", "/workspace/xtts_env/bin/python")
MUSE_PY = os.environ.get("MUSE_PY", "/workspace/musetalk_env/bin/python")

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
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if p.returncode != 0:
        tail = (p.stdout or "")[-8000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout or ""

# ---------------------------
# XTTS -> WAV (TTS env)
# ---------------------------
def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV

    if not os.path.isfile(speaker):
        raise RuntimeError(f"Missing speaker_wav: {speaker}")

    if not os.path.isfile(TTS_PY):
        raise RuntimeError(f"TTS_PY not found: {TTS_PY}")

    # tts_generate.py must exist inside the container at /app/tts_generate.py
    cmd = [
        TTS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav,
    ]
    _run(cmd)

# ---------------------------
# MuseTalk inference (Muse env)
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

    # Typical output location
    results_dir = os.path.join(MUSE_ROOT, "results", "v15")
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
        out_mp4 = _musetalk_infer(input_mp4=in_mp4, audio_wav=tts_wav)

        with open(out_mp4, "rb") as f:
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

        # mode can be in: input.mode OR input.ping (legacy)
        mode = str(inp.get("mode") or inp.get("ping") or "").strip().lower()

        if mode in ("debug", "echo"):
            return {
                "ok": True,
                "msg": "ECHO_OK",
                "paths": {
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "TTS_PY": TTS_PY,
                    "MUSE_PY": MUSE_PY,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                },
                "checks": {
                    "muse_root_exists": os.path.isdir(MUSE_ROOT),
                    "voices_dir_exists": os.path.isdir(VOICES_DIR),
                    "tts_py_exists": os.path.isfile(TTS_PY),
                    "muse_py_exists": os.path.isfile(MUSE_PY),
                    "female_ref_exists": os.path.isfile(FEMALE_REF_WAV),
                    "male_ref_exists": os.path.isfile(MALE_REF_WAV),
                },
            }

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})
