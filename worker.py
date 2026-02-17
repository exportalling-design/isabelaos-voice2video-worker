# /app/worker.py
import os
import gc
import time
import base64
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict

import runpod

# ----------------------------
# ENV hardening
# ----------------------------
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYS_PY = "/usr/local/bin/python3"

# ----------------------------
# Helpers: filesystem + alias mounts
# ----------------------------
def _exists_file(p: str) -> bool:
    return bool(p) and os.path.isfile(p)

def _exists_dir(p: str) -> bool:
    return bool(p) and os.path.isdir(p)

def _require_file(path: str, label: str):
    if not _exists_file(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _require_dir(path: str, label: str):
    if not _exists_dir(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _try_symlink(src: str, dst: str):
    """
    Create symlink dst -> src (dst path becomes alias for src)
    Safe: if dst exists, do nothing.
    """
    try:
        if os.path.exists(dst) or os.path.islink(dst):
            return
        parent = os.path.dirname(dst.rstrip("/"))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        os.symlink(src, dst)
    except Exception:
        # Don't crash worker for symlink issues; we will fall back to detection.
        pass

def _auto_alias_volume():
    """
    Many RunPod setups mount the Network Volume at /workspace (pod),
    while serverless examples often use /runpod-volume.
    We keep YOUR existing /runpod-volume logic by aliasing:
      - If /runpod-volume doesn't exist but /workspace does -> symlink /runpod-volume -> /workspace
      - If /runpod-volume exists but key dirs are missing and /workspace has them -> symlink per-dir
    """
    ws = "/workspace"
    rp = "/runpod-volume"

    # If /runpod-volume doesn't exist, alias whole mount
    if (not os.path.exists(rp)) and os.path.isdir(ws):
        _try_symlink(ws, rp)
        return

    # If /runpod-volume exists but looks "empty" for our needs, alias specific folders from /workspace
    if os.path.isdir(ws) and os.path.isdir(rp):
        for name in ("xtts_env", "xtts_env_persist", "voices", "MuseTalk", "musetalk_ok", "musetalk_ok_persist"):
            src = os.path.join(ws, name)
            dst = os.path.join(rp, name)
            if os.path.isdir(src) and (not os.path.exists(dst)):
                _try_symlink(src, dst)

_auto_alias_volume()

def _detect_base() -> str:
    """
    Keep default preference for /runpod-volume (YOUR existing logic),
    but if it's missing or unusable, fall back to /workspace.
    """
    rp_env = (os.environ.get("RUNPOD_VOLUME_PATH") or "").strip()
    if rp_env and os.path.isdir(rp_env):
        return rp_env

    # Prefer /runpod-volume if it exists (and after aliasing it might point to /workspace)
    if os.path.isdir("/runpod-volume"):
        return "/runpod-volume"

    if os.path.isdir("/workspace"):
        return "/workspace"

    return "/"

BASE = _detect_base()

# ----------------------------
# Paths (defaults)
# ----------------------------
# You can override these via endpoint env vars if you want, but not required.
MUSE_ROOT   = (os.environ.get("MUSE_ROOT") or f"{BASE}/MuseTalk").strip()
VOICES_DIR  = (os.environ.get("VOICES_DIR") or f"{BASE}/voices").strip()

# XTTS env + python inside the volume
XTTS_ENV_DIR = (os.environ.get("XTTS_ENV_DIR") or f"{BASE}/xtts_env").strip()
XTTS_PY      = (os.environ.get("XTTS_PY") or f"{XTTS_ENV_DIR}/bin/python").strip()

# Optional: reference wavs for cloning voice
FEMALE_REF_WAV = (os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav").strip()
MALE_REF_WAV   = (os.environ.get("MALE_REF_WAV") or f"{VOICES_DIR}/male_ref.wav").strip()

# If you actually use musetalk_ok folder name, allow auto-fallback
def _resolve_muse_root() -> str:
    # if provided exists, use it
    if os.path.isdir(MUSE_ROOT):
        return MUSE_ROOT
    # common alternatives inside same base
    alt1 = f"{BASE}/musetalk_ok"
    alt2 = f"{BASE}/MuseTalk"
    if os.path.isdir(alt1):
        return alt1
    if os.path.isdir(alt2):
        return alt2
    return MUSE_ROOT

MUSE_ROOT = _resolve_muse_root()

# ----------------------------
# Exec helpers
# ----------------------------
def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass

def _decode_b64(s: str) -> bytes:
    s = str(s).strip()
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()
    s = "".join(s.split()).replace("-", "+").replace("_", "/")
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad
    return base64.b64decode(s, validate=True)

def _b64_to_file(b64: str, out_path: str):
    raw = _decode_b64(b64)
    with open(out_path, "wb") as f:
        f.write(raw)

def _download_to_file(url: str, out_path: str):
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())

def _clean_env(extra: Dict[str, str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    if extra:
        env.update(extra)
    return env

def _run(cmd: list, cwd: str = None, env: Dict[str, str] = None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env if env is not None else _clean_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if p.returncode != 0:
        tail = (p.stdout or "")[-12000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout or ""

# ----------------------------
# XTTS (Coqui) wav generation via separate script
# ----------------------------
def _tts_make_wav_xtts(text: str, lang: str, out_wav: str, speaker_wav: str):
    _require_file(XTTS_PY, "XTTS_PY (xtts_env python)")
    _require_file("/app/xtts_generate.py", "xtts_generate.py")

    # speaker wav is optional, but if path is provided it must exist
    if speaker_wav:
        _require_file(speaker_wav, "speaker_wav")

    cmd = [
        XTTS_PY, "-u", "/app/xtts_generate.py",
        "--text", text,
        "--lang", lang,
        "--out_wav", out_wav,
    ]
    if speaker_wav:
        cmd += ["--speaker_wav", speaker_wav]

    _run(cmd, env=_clean_env({
        # auto-accept
        "COQUI_TOS_AGREED": "1",
    }))

# ----------------------------
# MuseTalk lip-sync
# ----------------------------
def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    _require_dir(MUSE_ROOT, "MUSE_ROOT (MuseTalk folder)")

    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    cmd = [
        SYS_PY, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
        "--bbox_shift", "0",
        "--use_float16"
    ]
    _run(cmd, cwd=MUSE_ROOT, env=_clean_env())

    # Some installs output in results/v15 (yours did)
    results_dir = os.path.join(MUSE_ROOT, "results", "v15")
    if not os.path.isdir(results_dir):
        # fallback: newest under results/
        results_root = os.path.join(MUSE_ROOT, "results")
        _require_dir(results_root, "MuseTalk results root")
        candidates = []
        for root, _, files in os.walk(results_root):
            for f in files:
                if f.endswith(".mp4"):
                    candidates.append(os.path.join(root, f))
        if not candidates:
            raise RuntimeError("MuseTalk no produjo mp4 (no mp4 found in results)")
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    mp4s = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".mp4")]
    if not mp4s:
        raise RuntimeError("MuseTalk no produjo mp4")
    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]

# ----------------------------
# Main mode
# ----------------------------
def voice_to_video(inp: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    text = str(inp.get("text") or "").strip()
    if not text:
        raise RuntimeError("Falta text")

    lang = str(inp.get("lang") or "es").strip().lower()
    if lang not in ("es", "en"):
        lang = "es"

    voice = str(inp.get("voice") or "female").strip().lower()
    if voice not in ("female", "male"):
        voice = "female"

    # speaker wav selection (you can also pass speaker_wav directly)
    speaker_wav = str(inp.get("speaker_wav") or "").strip()
    if not speaker_wav:
        if voice == "female" and _exists_file(FEMALE_REF_WAV):
            speaker_wav = FEMALE_REF_WAV
        elif voice == "male" and _exists_file(MALE_REF_WAV):
            speaker_wav = MALE_REF_WAV
        else:
            # allow empty (XTTS can still run depending on your xtts_generate implementation)
            speaker_wav = ""

    video_b64 = inp.get("video_b64") or inp.get("video")
    video_url = str(inp.get("video_url") or inp.get("videoUrl") or "").strip()
    if not video_b64 and not video_url:
        raise RuntimeError("Falta video_b64 o video_url")

    with tempfile.TemporaryDirectory() as td:
        in_mp4  = os.path.join(td, "in.mp4")
        tts_wav = os.path.join(td, "tts.wav")

        if video_url:
            _download_to_file(video_url, in_mp4)
        else:
            _b64_to_file(str(video_b64), in_mp4)

        _tts_make_wav_xtts(text=text, lang=lang, out_wav=tts_wav, speaker_wav=speaker_wav)
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
        "paths": {
            "MUSE_ROOT": MUSE_ROOT,
            "VOICES_DIR": VOICES_DIR,
            "XTTS_ENV_DIR": XTTS_ENV_DIR,
            "XTTS_PY": XTTS_PY,
        },
        "tts": {
            "engine": "xtts",
            "voice": voice,
            "lang": lang,
            "speaker_wav": speaker_wav,
            "tos": os.environ.get("COQUI_TOS_AGREED", ""),
        }
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode in ("echo", "debug"):
            return {
                "ok": True,
                "msg": "ECHO_OK",
                "base": BASE,
                "python": SYS_PY,
                "env": {
                    "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
                    "RUNPOD_VOLUME_PATH": os.environ.get("RUNPOD_VOLUME_PATH"),
                    "PYTHONPATH": os.environ.get("PYTHONPATH"),
                    "PYTHONHOME": os.environ.get("PYTHONHOME"),
                    "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
                },
                "paths": {
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "XTTS_ENV_DIR": XTTS_ENV_DIR,
                    "XTTS_PY": XTTS_PY,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                },
                "checks": {
                    "workspace_exists": os.path.isdir("/workspace"),
                    "runpod_volume_exists": os.path.isdir("/runpod-volume"),
                    "muse_root_exists": os.path.isdir(MUSE_ROOT),
                    "xtts_py_exists": os.path.isfile(XTTS_PY),
                    "female_ref_exists": os.path.isfile(FEMALE_REF_WAV),
                    "male_ref_exists": os.path.isfile(MALE_REF_WAV),
                },
                "ls_hint": {
                    "/workspace": sorted(os.listdir("/workspace"))[:50] if os.path.isdir("/workspace") else None,
                    "/runpod-volume": sorted(os.listdir("/runpod-volume"))[:50] if os.path.isdir("/runpod-volume") else None,
                }
            }

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})
