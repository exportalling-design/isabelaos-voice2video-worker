# /app/worker.py
import os
import gc
import time
import base64
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, Tuple, List

import runpod

# --- harden global env ---
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYS_PY = os.environ.get("SYS_PY", "/usr/local/bin/python3")


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

# --- paths (volume) ---
VOICES_DIR = os.environ.get("VOICES_DIR") or f"{BASE}/voices"
FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV = os.environ.get("MALE_REF_WAV") or f"{VOICES_DIR}/male_ref.wav"

# MuseTalk
MUSE_ROOT = os.environ.get("MUSE_ROOT") or f"{BASE}/MuseTalk"
MUSE_ENV_DIR = os.environ.get("MUSE_ENV_DIR") or f"{BASE}/musetalk_ok"

# XTTS (Coqui)
XTTS_ENV_DIR = os.environ.get("XTTS_ENV_DIR") or f"{BASE}/xtts_env"
XTTS_MODEL_NAME = os.environ.get(
    "XTTS_MODEL_NAME",
    "tts_models/multilingual/multi-dataset/xtts_v2"
)
TTS_HOME = os.environ.get("TTS_HOME") or f"{BASE}/tts_cache"


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


def _safe_listdir(path: str, max_items: int = 200) -> List[Dict[str, Any]]:
    try:
        items = []
        for name in sorted(os.listdir(path))[:max_items]:
            p = os.path.join(path, name)
            items.append({
                "name": name,
                "is_dir": os.path.isdir(p),
                "is_file": os.path.isfile(p),
                "size": os.path.getsize(p) if os.path.isfile(p) else None
            })
        return items
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]


def _find_site_packages(env_dir: str) -> Optional[str]:
    """
    Finds env_dir/lib/pythonX.Y/site-packages (works even if env_dir/bin/python is missing).
    """
    lib_dir = os.path.join(env_dir, "lib")
    if not os.path.isdir(lib_dir):
        return None
    try:
        for pyver in sorted(os.listdir(lib_dir), reverse=True):
            sp = os.path.join(lib_dir, pyver, "site-packages")
            if os.path.isdir(sp):
                return sp
    except Exception:
        return None
    return None


def _pick_python(preferred: str, env_dir: str) -> Tuple[str, Optional[str]]:
    """
    If preferred exists, use it.
    If not, fallback to SYS_PY but return site-packages from env_dir to inject via PYTHONPATH.
    """
    if preferred and os.path.isfile(preferred):
        return preferred, None
    # try common links
    for cand in (preferred, preferred + "3", preferred + "3.11", preferred + "3.10"):
        if cand and os.path.isfile(cand):
            return cand, None
    sp = _find_site_packages(env_dir)
    return SYS_PY, sp


# Preferred venv pythons (may be missing in serverless images)
XTTS_PY_PREF = os.environ.get("XTTS_PY") or f"{XTTS_ENV_DIR}/bin/python"
MUSE_PY_PREF = os.environ.get("MUSE_PY") or f"{MUSE_ENV_DIR}/bin/python"

XTTS_PY, XTTS_SITE = _pick_python(XTTS_PY_PREF, XTTS_ENV_DIR)
MUSE_PY, MUSE_SITE = _pick_python(MUSE_PY_PREF, MUSE_ENV_DIR)


def _clean_env(extra: Dict[str, str] = None, add_py_path: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"

    # ✅ Auto-accept Coqui TOS in non-interactive serverless
    env["COQUI_TOS_AGREED"] = env.get("COQUI_TOS_AGREED", "1")

    # ✅ Make model/cache persistent (volume)
    env["TTS_HOME"] = env.get("TTS_HOME", TTS_HOME)

    if add_py_path:
        # Prepend site-packages so SYS_PY can import from the volume env
        env["PYTHONPATH"] = add_py_path

    if extra:
        env.update(extra)
    return env


def _run(cmd: list, cwd: str = None, env: Dict[str, str] = None) -> str:
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


def _ensure_ffmpeg():
    # MuseTalk needs ffmpeg. If not present, fail with clear message.
    try:
        _run(["bash", "-lc", "ffmpeg -version | head -n 1"], env=_clean_env())
    except Exception:
        raise RuntimeError("ffmpeg not found in image. Install ffmpeg in the serverless image.")


def _tts_make_wav_xtts(text: str, lang: str, out_wav: str, speaker_wav: str):
    _require_file(speaker_wav, "speaker_wav")

    # If we have site-packages, inject so SYS_PY can import TTS from /runpod-volume/xtts_env
    env = _clean_env(
        extra={
            "XTTS_MODEL_NAME": XTTS_MODEL_NAME,
        },
        add_py_path=XTTS_SITE
    )

    cmd = [
        XTTS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker_wav,
        "--out_wav", out_wav,
        "--model_name", XTTS_MODEL_NAME,
    ]
    _run(cmd, env=env)


def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    _require_dir(MUSE_ROOT, "MUSE_ROOT (MuseTalk folder)")
    _ensure_ffmpeg()

    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    env = _clean_env(add_py_path=MUSE_SITE)

    cmd = [
        MUSE_PY, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
        "--bbox_shift", "0",
        "--use_float16"
    ]
    _run(cmd, cwd=MUSE_ROOT, env=env)

    results_dir = os.path.join(MUSE_ROOT, "results", "v15")
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
    video_url = str(inp.get("video_url") or inp.get("videoUrl") or "").strip()
    if not video_b64 and not video_url:
        raise RuntimeError("Falta video_b64 o video_url")

    speaker_wav = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker_wav, f"{voice}_ref_wav")

    with tempfile.TemporaryDirectory() as td:
        in_mp4 = os.path.join(td, "in.mp4")
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
            "XTTS_ENV_DIR": XTTS_ENV_DIR,
            "XTTS_PY_PREF": XTTS_PY_PREF,
            "XTTS_PY_USED": XTTS_PY,
            "XTTS_SITE": XTTS_SITE,
            "MUSE_ENV_DIR": MUSE_ENV_DIR,
            "MUSE_PY_PREF": MUSE_PY_PREF,
            "MUSE_PY_USED": MUSE_PY,
            "MUSE_SITE": MUSE_SITE,
            "TTS_HOME": TTS_HOME,
        },
        "tts": {
            "engine": "xtts",
            "model_name": XTTS_MODEL_NAME,
            "voice": voice,
            "lang": lang,
            "speaker_wav": speaker_wav,
            "coqui_tos_agreed": os.environ.get("COQUI_TOS_AGREED", None),
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
                    "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
                    "TTS_HOME": os.environ.get("TTS_HOME"),
                    "PYTHONPATH": os.environ.get("PYTHONPATH"),
                    "PYTHONHOME": os.environ.get("PYTHONHOME"),
                    "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
                },
                "checks": {
                    "runpod_volume_exists": os.path.isdir("/runpod-volume"),
                    "voices_dir_exists": _exists_dir(VOICES_DIR),
                    "female_ref_exists": _exists_file(FEMALE_REF_WAV),
                    "male_ref_exists": _exists_file(MALE_REF_WAV),
                    "muse_root_exists": _exists_dir(MUSE_ROOT),
                    "xtts_env_exists": _exists_dir(XTTS_ENV_DIR),
                    "xtts_py_exists": _exists_file(XTTS_PY_PREF),
                },
                "paths": {
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "XTTS_ENV_DIR": XTTS_ENV_DIR,
                    "XTTS_PY": XTTS_PY_PREF,
                    "XTTS_PY_USED": XTTS_PY,
                    "XTTS_SITE": XTTS_SITE,
                    "MUSE_ENV_DIR": MUSE_ENV_DIR,
                    "MUSE_PY": MUSE_PY_PREF,
                    "MUSE_PY_USED": MUSE_PY,
                    "MUSE_SITE": MUSE_SITE,
                    "TTS_HOME": TTS_HOME,
                },
            }

        if mode in ("ls", "list"):
            return {
                "ok": True,
                "base": BASE,
                "ls_base": _safe_listdir(BASE),
                "ls_runpod_volume": _safe_listdir("/runpod-volume"),
                "exists": {
                    "runpod_volume": os.path.isdir("/runpod-volume"),
                    "voices_dir": os.path.isdir(VOICES_DIR),
                    "MuseTalk_dir": os.path.isdir(f"{BASE}/MuseTalk"),
                    "xtts_env_dir": os.path.isdir(f"{BASE}/xtts_env"),
                    "musetalk_ok_dir": os.path.isdir(f"{BASE}/musetalk_ok"),
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
