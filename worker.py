
# /app/worker.py
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

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _detect_base() -> str:
    candidates = []
    rp = (os.environ.get("RUNPOD_VOLUME_PATH") or "").strip()
    if rp:
        candidates.append(rp)
    for k in ("VOLUME_PATH", "BASE"):
        v = (os.environ.get(k) or "").strip()
        if v:
            candidates.append(v)
    candidates += ["/runpod-volume", "/workspace", "/mnt", "/data", "/volume", "/workspace/runpod-volume"]
    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mp = parts[1].strip()
                    if mp and mp not in candidates:
                        candidates.append(mp)
    except Exception:
        pass

    for base in [c for c in candidates if c]:
        if os.path.isdir(os.path.join(base, "MuseTalk")) and os.path.isdir(os.path.join(base, "voices")):
            return base
    return rp or "/runpod-volume"

BASE = _detect_base()

MUSE_ROOT   = os.environ.get("MUSE_ROOT")   or f"{BASE}/MuseTalk"
VOICES_DIR  = os.environ.get("VOICES_DIR")  or f"{BASE}/voices"

FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV   = os.environ.get("MALE_REF_WAV")   or f"{VOICES_DIR}/male_ref.wav"

TTS_PY  = os.environ.get("TTS_PY")  or f"{BASE}/xtts_env/bin/python"
MUSE_PY = os.environ.get("MUSE_PY") or f"{BASE}/musetalk_ok/bin/python"

def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass

def _decode_b64(s: str) -> bytes:
    if not s:
        raise ValueError("b64 vacío")
    s = str(s).strip()
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()
    s = "".join(s.split())
    s = s.replace("-", "+").replace("_", "/")
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"b64 inválido: {e}")

def _b64_to_file(b64: str, out_path: str):
    raw = _decode_b64(b64)
    with open(out_path, "wb") as f:
        f.write(raw)

def _download_to_file(url: str, out_path: str):
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())

def _run(cmd: list, cwd: str = None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        tail = (p.stdout or "")[-8000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout or ""

def _require_file(path: str, label: str):
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _require_dir(path: str, label: str):
    if not path or not os.path.isdir(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")
    _require_file(TTS_PY, "TTS_PY (xtts_env python)")
    cmd = [TTS_PY, "-u", "/app/tts_generate.py", "--text", text, "--lang", lang, "--speaker_wav", speaker, "--out_wav", out_wav]
    _run(cmd)

def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    _require_dir(MUSE_ROOT, "MUSE_ROOT (MuseTalk folder)")
    _require_file(MUSE_PY, "MUSE_PY (musetalk_ok python)")

    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"])
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"])

    cmd = [MUSE_PY, "-u", "scripts/inference.py", "--inference_config", "inference_config.json", "--bbox_shift", "0", "--use_float16"]
    _run(cmd, cwd=MUSE_ROOT)

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

    video_b64 = inp.get("video_b64") or inp.get("video")
    video_url = str(inp.get("video_url") or "").strip()
    if not video_b64 and not video_url:
        raise RuntimeError("Falta video_b64 o video_url")

    with tempfile.TemporaryDirectory() as td:
        in_mp4  = os.path.join(td, "in.mp4")
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
        "elapsed_s": round(time.time() - t0, 3),
        "video_b64": base64.b64encode(mp4_bytes).decode("utf-8"),
        "video_mime": "video/mp4",
        "base": BASE,
    }

def _safe_listdir(path: str, max_items: int = 200):
    try:
        if not os.path.isdir(path):
            return {"ok": False, "error": "not_a_dir", "path": path}
        items = sorted(os.listdir(path))[:max_items]
        return {"ok": True, "path": path, "items": items}
    except Exception as e:
        return {"ok": False, "error": str(e), "path": path}

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or inp.get("ping") or "").strip().lower()

        if mode in ("echo", "debug"):
            env_dump = {k: v for (k, v) in os.environ.items() if k.startswith(("MUSE","TTS","BASE","FEMALE","MALE","RUNPOD","VOLUME","VOICES"))}
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
                    "path_env": os.environ.get("PATH", ""),
                },
                "env_dump": env_dump,
            }

        if mode in ("ls", "list"):
            # lista los bins reales para dejarlo cerrado
            return {
                "ok": True,
                "base": BASE,
                "want": {
                    "TTS_PY": TTS_PY,
                    "MUSE_PY": MUSE_PY,
                },
                "list": {
                    "xtts_bin": _safe_listdir(f"{BASE}/xtts_env/bin"),
                    "musetalk_bin": _safe_listdir(f"{BASE}/musetalk_ok/bin"),
                }
            }

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|ls|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})
