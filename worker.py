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
import shutil
import glob
from typing import Any, Dict, Optional, Tuple

import runpod

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ----------------------------
# Helpers: paths & detection
# ----------------------------
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

    # also add mount points (best effort)
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

MUSE_ROOT  = os.environ.get("MUSE_ROOT")  or f"{BASE}/MuseTalk"
VOICES_DIR = os.environ.get("VOICES_DIR") or f"{BASE}/voices"

FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV   = os.environ.get("MALE_REF_WAV")   or f"{VOICES_DIR}/male_ref.wav"

# venv roots in volume (packages live here)
XTTS_ENV_DIR   = os.environ.get("XTTS_ENV_DIR")   or f"{BASE}/xtts_env"
MUSE_ENV_DIR   = os.environ.get("MUSE_ENV_DIR")   or f"{BASE}/musetalk_ok"

# "preferred" python inside venv (may NOT be runnable on serverless)
TTS_PY  = os.environ.get("TTS_PY")  or f"{XTTS_ENV_DIR}/bin/python"
MUSE_PY = os.environ.get("MUSE_PY") or f"{MUSE_ENV_DIR}/bin/python"

# sometimes you want the tts CLI too
TTS_BIN = os.environ.get("TTS_BIN") or f"{XTTS_ENV_DIR}/bin/tts"


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


def _run(cmd: list, cwd: str = None, env: Dict[str, str] = None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
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


# ----------------------------
# Key fix: python picker + PYTHONPATH
# ----------------------------
def _find_site_packages(env_dir: str) -> Optional[str]:
    # try common patterns
    candidates = glob.glob(os.path.join(env_dir, "lib", "python*", "site-packages"))
    candidates += glob.glob(os.path.join(env_dir, "Lib", "site-packages"))  # windows-ish
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _python_runnable(py_path: str) -> Tuple[bool, str]:
    try:
        out = _run([py_path, "-V"])
        return True, out.strip()
    except Exception as e:
        return False, str(e)


def _pick_python(venv_python: str, venv_dir: str) -> Dict[str, Any]:
    """
    In serverless, venv python binaries copied from another image can be UNRUNNABLE.
    So:
      1) Try venv_python
      2) fallback to system python3 (PATH)
      3) Provide PYTHONPATH to venv site-packages so imports still work
    """
    tested = []
    ok, out = _python_runnable(venv_python)
    tested.append({"path": venv_python, "ok": ok, "out": out})

    if ok:
        sys_py = venv_python
    else:
        sys_py = shutil.which("python3") or shutil.which("python") or ""
        if not sys_py:
            return {
                "ok": False,
                "why": "no python3/python in PATH and venv python not runnable",
                "tested": tested,
                "python": venv_python,
                "site_packages": _find_site_packages(venv_dir),
            }
        ok2, out2 = _python_runnable(sys_py)
        tested.append({"path": sys_py, "ok": ok2, "out": out2})
        if not ok2:
            return {
                "ok": False,
                "why": "system python exists but not runnable",
                "tested": tested,
                "python": sys_py,
                "site_packages": _find_site_packages(venv_dir),
            }

    sp = _find_site_packages(venv_dir)
    return {
        "ok": True,
        "python": sys_py,
        "site_packages": sp,
        "tested": tested,
    }


def _env_with_pythonpath(site_packages: Optional[str]) -> Dict[str, str]:
    env = dict(os.environ)
    if site_packages:
        prev = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = f"{site_packages}:{prev}" if prev else site_packages
    return env


# Pickers (computed once per container start)
PICK_TTS  = _pick_python(TTS_PY,  XTTS_ENV_DIR)
PICK_MUSE = _pick_python(MUSE_PY, MUSE_ENV_DIR)

TTS_SYS_PY  = PICK_TTS.get("python") if PICK_TTS.get("ok") else (shutil.which("python3") or "python3")
MUSE_SYS_PY = PICK_MUSE.get("python") if PICK_MUSE.get("ok") else (shutil.which("python3") or "python3")

TTS_ENV  = _env_with_pythonpath(PICK_TTS.get("site_packages"))
MUSE_ENV = _env_with_pythonpath(PICK_MUSE.get("site_packages"))


# ----------------------------
# Pipeline
# ----------------------------
def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")
    _require_file("/app/tts_generate.py", "tts_generate.py")

    # IMPORTANT: use system python (serverless) + PYTHONPATH pointing to xtts_env site-packages
    cmd = [TTS_SYS_PY, "-u", "/app/tts_generate.py",
           "--text", text,
           "--lang", lang,
           "--speaker_wav", speaker,
           "--out_wav", out_wav]
    _run(cmd, env=TTS_ENV)


def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    _require_dir(MUSE_ROOT, "MUSE_ROOT (MuseTalk folder)")
    _require_file(audio_wav, "audio_wav")
    _require_file(input_mp4, "input_mp4")

    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"])
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"])

    # IMPORTANT: use system python + PYTHONPATH pointing to musetalk_ok site-packages
    cmd = [MUSE_SYS_PY, "-u", "scripts/inference.py",
           "--inference_config", "inference_config.json",
           "--bbox_shift", "0",
           "--use_float16"]
    _run(cmd, cwd=MUSE_ROOT, env=MUSE_ENV)

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
        "python_used": {
            "tts": TTS_SYS_PY,
            "muse": MUSE_SYS_PY,
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
            env_dump = {k: v for (k, v) in os.environ.items()
                        if k.startswith(("MUSE","TTS","BASE","FEMALE","MALE","RUNPOD","VOLUME","VOICES","XTTS"))}
            # mount hints
            proc_mounts = ""
            try:
                with open("/proc/mounts","r") as f:
                    proc_mounts = f.read()
            except Exception:
                pass

            # system python info
            sys_py = shutil.which("python3") or shutil.which("python") or ""
            sys_py_ver = ""
            if sys_py:
                try:
                    sys_py_ver = _run([sys_py, "-V"]).strip()
                except Exception as e:
                    sys_py_ver = f"[FAILED] {e}"

            return {
                "ok": True,
                "msg": "ECHO_OK",
                "base": BASE,
                "paths": {
                    "BASE": BASE,
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "XTTS_ENV_DIR": XTTS_ENV_DIR,
                    "MUSE_ENV_DIR": MUSE_ENV_DIR,
                    "TTS_PY_PREF": TTS_PY,
                    "MUSE_PY_PREF": MUSE_PY,
                    "TTS_BIN": TTS_BIN,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                },
                "checks": {
                    "base_exists": os.path.isdir(BASE),
                    "muse_root_exists": os.path.isdir(MUSE_ROOT),
                    "voices_dir_exists": os.path.isdir(VOICES_DIR),
                    "female_ref_exists": os.path.isfile(FEMALE_REF_WAV),
                    "male_ref_exists": os.path.isfile(MALE_REF_WAV),
                    "xtts_env_exists": os.path.isdir(XTTS_ENV_DIR),
                    "muse_env_exists": os.path.isdir(MUSE_ENV_DIR),
                    "sys_python3": sys_py,
                    "sys_python3_ver": sys_py_ver,
                    "pick_tts": PICK_TTS,
                    "pick_muse": PICK_MUSE,
                },
                "python_used": {
                    "tts": TTS_SYS_PY,
                    "muse": MUSE_SYS_PY,
                    "tts_site_packages": PICK_TTS.get("site_packages"),
                    "muse_site_packages": PICK_MUSE.get("site_packages"),
                },
                "mount_hint": {
                    "proc_mounts_has_runpod_volume": ("/runpod-volume" in proc_mounts),
                    "proc_mounts_has_workspace": ("/workspace" in proc_mounts),
                },
                "env_dump": env_dump,
            }

        if mode in ("ls", "list"):
            return {
                "ok": True,
                "base": BASE,
                "want": {
                    "MUSE_ENV_DIR": MUSE_ENV_DIR,
                    "XTTS_ENV_DIR": XTTS_ENV_DIR,
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "TTS_PY_PREF": TTS_PY,
                    "MUSE_PY_PREF": MUSE_PY,
                },
                "list": {
                    "musetalk_bin": _safe_listdir(f"{MUSE_ENV_DIR}/bin"),
                    "xtts_bin": _safe_listdir(f"{XTTS_ENV_DIR}/bin"),
                    "musetalk_root": _safe_listdir(MUSE_ROOT),
                    "voices": _safe_listdir(VOICES_DIR),
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
