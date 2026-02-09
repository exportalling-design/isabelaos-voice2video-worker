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
from typing import Any, Dict, Optional, List

import runpod

# --- ENV hardening ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYS_PY = os.environ.get("SYS_PY", "/usr/local/bin/python3")

def _detect_base() -> str:
    candidates: List[str] = []
    rp = (os.environ.get("RUNPOD_VOLUME_PATH") or "").strip()
    if rp:
        candidates.append(rp)

    for k in ("VOLUME_PATH", "BASE"):
        v = (os.environ.get(k) or "").strip()
        if v:
            candidates.append(v)

    candidates += ["/runpod-volume", "/workspace", "/mnt", "/data", "/volume", "/workspace/runpod-volume"]

    # try mounts
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

    # pick one that has what we need
    for base in [c for c in candidates if c]:
        if os.path.isdir(os.path.join(base, "MuseTalk")) and os.path.isdir(os.path.join(base, "voices")):
            return base

    return rp or "/runpod-volume"

BASE = _detect_base()

MUSE_ROOT   = os.environ.get("MUSE_ROOT")   or f"{BASE}/MuseTalk"
VOICES_DIR  = os.environ.get("VOICES_DIR")  or f"{BASE}/voices"

FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV   = os.environ.get("MALE_REF_WAV")   or f"{VOICES_DIR}/male_ref.wav"

# Opcional: si ya tenés site-packages dentro del volumen, los agregamos por PYTHONPATH
MUSE_SITE = os.environ.get("MUSE_SITE") or f"{BASE}/musetalk_ok/lib/python3.11/site-packages"
TTS_SITE  = os.environ.get("TTS_SITE")  or f"{BASE}/xtts_env/lib/python3.11/site-packages"

def _exists_dir(p: str) -> bool:
    try:
        return bool(p) and os.path.isdir(p)
    except Exception:
        return False

def _exists_file(p: str) -> bool:
    try:
        return bool(p) and os.path.isfile(p)
    except Exception:
        return False

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

def _require_file(path: str, label: str):
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _require_dir(path: str, label: str):
    if not path or not os.path.isdir(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _build_env(extra_site: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)

    # IMPORTANT: En serverless, aunque el volumen tenga envs, NO uses bin/python del volumen.
    # Usamos SYS_PY siempre y sumamos site-packages por PYTHONPATH si existen.
    pieces = []

    # extra_site first
    if extra_site and _exists_dir(extra_site):
        pieces.append(extra_site)

    # keep existing PYTHONPATH
    if env.get("PYTHONPATH"):
        pieces.append(env["PYTHONPATH"])

    env["PYTHONPATH"] = ":".join([p for p in pieces if p])

    return env

def _run(cmd: list, cwd: str = None, env: Dict[str, str] = None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if p.returncode != 0:
        tail = (p.stdout or "")[-12000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout or ""

def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")

    # Si en tu volumen tenés TTS instalado en xtts_env site-packages, lo sumamos.
    env = _build_env(TTS_SITE if _exists_dir(TTS_SITE) else None)

    cmd = [
        SYS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav
    ]
    _run(cmd, env=env)

def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    _require_dir(MUSE_ROOT, "MUSE_ROOT (MuseTalk folder)")

    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    # copiar inputs
    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"])
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"])

    # MuseTalk deps: si las tenés en musetalk_ok site-packages, sumamos.
    env = _build_env(MUSE_SITE if _exists_dir(MUSE_SITE) else None)

    cmd = [
        SYS_PY, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
        "--bbox_shift", "0",
        "--use_float16"
    ]
    _run(cmd, cwd=MUSE_ROOT, env=env)

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
        "python": SYS_PY,
        "sites": {
            "MUSE_SITE_used": MUSE_SITE if _exists_dir(MUSE_SITE) else None,
            "TTS_SITE_used": TTS_SITE if _exists_dir(TTS_SITE) else None
        }
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
            return {
                "ok": True,
                "msg": "ECHO_OK",
                "base": BASE,
                "paths": {
                    "BASE": BASE,
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                    "SYS_PY": SYS_PY,
                    "MUSE_SITE": MUSE_SITE,
                    "TTS_SITE": TTS_SITE,
                },
                "checks": {
                    "base_exists": os.path.isdir(BASE),
                    "muse_root_exists": os.path.isdir(MUSE_ROOT),
                    "voices_dir_exists": os.path.isdir(VOICES_DIR),
                    "female_ref_exists": _exists_file(FEMALE_REF_WAV),
                    "male_ref_exists": _exists_file(MALE_REF_WAV),
                    "sys_py_exists": _exists_file(SYS_PY),
                    "muse_site_exists": _exists_dir(MUSE_SITE),
                    "tts_site_exists": _exists_dir(TTS_SITE),
                },
                "mount_hint": {
                    "proc_mounts_has_runpod_volume": ("runpod-volume" in open("/proc/mounts", "r").read()) if os.path.exists("/proc/mounts") else None
                }
            }

        if mode in ("ls", "list"):
            return {
                "ok": True,
                "base": BASE,
                "list": {
                    "musetalk_root": _safe_listdir(MUSE_ROOT),
                    "voices": _safe_listdir(VOICES_DIR),
                    "muse_site": _safe_listdir(MUSE_SITE) if _exists_dir(MUSE_SITE) else {"ok": False, "path": MUSE_SITE},
                    "tts_site": _safe_listdir(TTS_SITE) if _exists_dir(TTS_SITE) else {"ok": False, "path": TTS_SITE},
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
