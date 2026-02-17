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

# ✅ Mata contaminación global
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("COQUI_TOS_AGREED", "1")  # ✅ evita prompt y/n (cuando aplica)

SYS_PY = "/usr/local/bin/python3"


# ----------------------------
# Paths / Base detection
# ----------------------------
def _detect_base() -> str:
    rp = (os.environ.get("RUNPOD_VOLUME_PATH") or "").strip()
    if rp and os.path.isdir(rp):
        return rp
    if os.path.isdir("/runpod-volume"):
        return "/runpod-volume"
    if os.path.isdir("/workspace"):
        return "/workspace"
    return "/"


BASE = _detect_base()

VOICES_DIR = os.environ.get("VOICES_DIR") or f"{BASE}/voices"
FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV = os.environ.get("MALE_REF_WAV") or f"{VOICES_DIR}/male_ref.wav"


# ✅ FIX CRÍTICO: MUSE_ROOT SIEMPRE debe ser carpeta repo (no bin/python)
def _normalize_muse_root(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return ""
    # Si te pasaron ".../bin/python" o cualquier cosa bajo /bin/, subimos 2 niveles
    if p.endswith("/bin/python") or p.endswith("/bin/python3") or "/bin/" in p:
        p2 = os.path.dirname(os.path.dirname(p))  # quita /bin/...
        if os.path.isdir(p2):
            return p2
    return p


def _pick_muse_root(base: str) -> str:
    # 1) Env (normalizado)
    env_muse = _normalize_muse_root(os.environ.get("MUSE_ROOT") or "")
    if env_muse and os.path.isdir(env_muse):
        return env_muse

    # 2) Autodetect (tus nombres reales)
    candidates = [
        f"{base}/musetalk_ok",
        f"{base}/MuseTalk",
        f"{base}/MuseTalk_ok",
        f"{base}/musetalk_ok_persist",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c

    # 3) Default esperado (para error claro)
    return f"{base}/musetalk_ok"


MUSE_ROOT = _pick_muse_root(BASE)


# ----------------------------
# Helpers
# ----------------------------
def _exists_file(p: str) -> bool:
    return bool(p) and os.path.isfile(p)


def _require_file(path: str, label: str):
    if not _exists_file(path):
        raise RuntimeError(f"Missing {label}: {path}")


def _require_dir(path: str, label: str):
    if not path or not os.path.isdir(path):
        raise RuntimeError(f"Missing {label}: {path}")


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
    env.setdefault("COQUI_TOS_AGREED", "1")
    if extra:
        env.update(extra)
    return env


def _run(cmd: list, cwd: str = None, env: Dict[str, str] = None, stdin_text: str = None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env if env is not None else _clean_env(),
        input=stdin_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if p.returncode != 0:
        tail = (p.stdout or "")[-12000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout or ""


# ----------------------------
# XTTS (Coqui) -> WAV
# ----------------------------
def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")

    cmd = [
        SYS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav
    ]

    # ✅ por si acaso la lib intenta pedir input (serverless no tiene stdin interactivo)
    _run(cmd, env=_clean_env(), stdin_text="y\n")


# ----------------------------
# MuseTalk inference
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

    results_dir = os.path.join(MUSE_ROOT, "results", "v15")
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"No results dir: {results_dir}")

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

    voice = str(inp.get("voice") or "female").strip().lower()
    if voice not in ("female", "male"):
        voice = "female"

    lang = str(inp.get("lang") or "es").strip().lower()
    if lang not in ("es", "en"):
        lang = "es"

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
        "paths": {
            "MUSE_ROOT": MUSE_ROOT,
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
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
                "python": SYS_PY,
                "base": BASE,
                "env": {
                    "RUNPOD_VOLUME_PATH": os.environ.get("RUNPOD_VOLUME_PATH"),
                    "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
                },
                "checks": {
                    "muse_root_exists": os.path.isdir(MUSE_ROOT),
                    "voices_dir_exists": os.path.isdir(VOICES_DIR),
                    "female_ref_exists": _exists_file(FEMALE_REF_WAV),
                    "male_ref_exists": _exists_file(MALE_REF_WAV),
                },
                "paths": {
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
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
