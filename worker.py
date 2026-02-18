# /app/worker.py
import os
import gc
import time
import base64
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, List

import runpod

# ----------------------------
# Global env hardening
# ----------------------------
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("TTS_USE_GPU", "1")

WORKER_VERSION = "voice2video_v4_fix_musetalk_path_no_venv"

SYS_PY = "/usr/local/bin/python3"

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
MALE_REF_WAV   = os.environ.get("MALE_REF_WAV")   or f"{VOICES_DIR}/male_ref.wav"

# MuseTalk repo (tu caso real es volume_old/MuseTalk)
MUSE_REPO_DEFAULT = f"{BASE}/volume_old/MuseTalk"
MUSE_REPO = os.environ.get("MUSE_REPO") or MUSE_REPO_DEFAULT

# Config: en tu caso existe en el repo
MUSE_CONFIG_DEFAULT = f"{MUSE_REPO}/inference_config.json"
MUSE_CONFIG = os.environ.get("MUSE_CONFIG") or MUSE_CONFIG_DEFAULT

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

def _tail(s: str, n: int = 12000) -> str:
    s = s or ""
    return s[-n:]

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
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if extra:
        env.update(extra)
    return env

def _run(cmd: List[str], cwd: str = None, env: Dict[str, str] = None, stdin_text: str = None) -> str:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env if env is not None else _clean_env(),
        input=stdin_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    out = p.stdout or ""
    if p.returncode != 0:
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{_tail(out)}\n")
    return out

# ----------------------------
# XTTS
# ----------------------------
def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str) -> Dict[str, Any]:
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")

    # Intento GPU primero, si truena (ECC) reintenta CPU
    cmd_gpu = [
        SYS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav,
    ]
    try:
        out = _run(cmd_gpu, env=_clean_env(), stdin_text="y\n")
        return {"ok": True, "used_gpu": True, "log_tail": _tail(out, 2000)}
    except Exception as e:
        # fallback CPU
        cmd_cpu = cmd_gpu + ["--force_cpu"]
        out2 = _run(cmd_cpu, env=_clean_env({"TTS_USE_GPU": "0"}), stdin_text="y\n")
        return {"ok": True, "used_gpu": False, "log_tail": _tail(out2, 2000), "gpu_error": str(e)}

# ----------------------------
# MuseTalk
# ----------------------------
def _musetalk_infer(repo_root: str, config_json: str, input_mp4: str, audio_wav: str) -> Dict[str, Any]:
    _require_dir(repo_root, "MUSE_REPO")
    _require_file(f"{repo_root}/scripts/inference.py", "MuseTalk scripts/inference.py")
    _require_file(config_json, "MUSE_CONFIG")

    # MuseTalk lee inputs/ por defecto en su repo
    inputs_dir = os.path.join(repo_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)
    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    # 🔥 clave: PYTHONPATH apunta al repo para que "import musetalk" exista
    env = _clean_env({
        "PYTHONPATH": repo_root,
    })

    cmd = [
        SYS_PY, "-u", "scripts/inference.py",
        "--inference_config", os.path.basename(config_json),
        "--bbox_shift", "0",
        "--use_float16"
    ]

    # Asegura que inference_config.json esté en cwd con ese nombre
    # (sin copiar repos enormes: solo link/copy chiquito)
    cfg_local = os.path.join(repo_root, "inference_config.json")
    if os.path.abspath(config_json) != os.path.abspath(cfg_local):
        _run(["bash", "-lc", f"cp -f '{config_json}' '{cfg_local}'"], env=_clean_env())

    out = _run(cmd, cwd=repo_root, env=env)

    # Busca mp4 más reciente en results
    results_dir = os.path.join(repo_root, "results")
    if not os.path.isdir(results_dir):
        raise RuntimeError("MuseTalk no creó results/ \n" + _tail(out))

    mp4s: List[str] = []
    for root, _, files in os.walk(results_dir):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                mp4s.append(os.path.join(root, fn))

    if not mp4s:
        raise RuntimeError("MuseTalk no produjo mp4\n" + _tail(out))

    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return {"out_mp4_path": mp4s[0], "log_tail": _tail(out, 3000)}

# ----------------------------
# Modes
# ----------------------------
def mode_echo() -> Dict[str, Any]:
    return {
        "ok": True,
        "msg": "ECHO_OK",
        "worker_version": WORKER_VERSION,
        "base": BASE,
        "python": SYS_PY,
        "env": {
            "RUNPOD_VOLUME_PATH": os.environ.get("RUNPOD_VOLUME_PATH"),
            "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
            "TTS_USE_GPU": os.environ.get("TTS_USE_GPU"),
        },
        "checks": {
            "voices_dir_exists": _exists_dir(VOICES_DIR),
            "female_ref_exists": _exists_file(FEMALE_REF_WAV),
            "male_ref_exists": _exists_file(MALE_REF_WAV),
            "muse_repo_exists": _exists_dir(MUSE_REPO),
            "muse_config_exists": _exists_file(MUSE_CONFIG),
            "muse_scripts_inference_exists": _exists_file(f"{MUSE_REPO}/scripts/inference.py"),
        },
        "paths": {
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "MUSE_REPO": MUSE_REPO,
            "MUSE_CONFIG": MUSE_CONFIG,
        },
    }

def mode_muse_debug() -> Dict[str, Any]:
    _require_dir(MUSE_REPO, "MUSE_REPO")
    _require_file(f"{MUSE_REPO}/scripts/inference.py", "scripts/inference.py")
    _require_file(MUSE_CONFIG, "MUSE_CONFIG")

    # verifica imports con el python del container + PYTHONPATH al repo
    env = _clean_env({"PYTHONPATH": MUSE_REPO})
    out = _run([SYS_PY, "-c", "import cv2, diffusers, mmpose; import musetalk; print('IMPORTS_OK')"], env=env)
    return {
        "ok": True,
        "msg": "MUSE_DEBUG_OK",
        "worker_version": WORKER_VERSION,
        "python": SYS_PY,
        "repo": MUSE_REPO,
        "config": MUSE_CONFIG,
        "imports": _tail(out, 2000),
    }

def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
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

    _require_dir(MUSE_REPO, "MUSE_REPO")
    _require_file(MUSE_CONFIG, "MUSE_CONFIG")

    with tempfile.TemporaryDirectory() as td:
        in_mp4  = os.path.join(td, "in.mp4")
        tts_wav = os.path.join(td, "tts.wav")

        if video_url:
            _download_to_file(video_url, in_mp4)
        else:
            _b64_to_file(str(video_b64), in_mp4)

        tts_info = _tts_make_wav(text=text, voice=voice, lang=lang, out_wav=tts_wav)
        muse_info = _musetalk_infer(repo_root=MUSE_REPO, config_json=MUSE_CONFIG, input_mp4=in_mp4, audio_wav=tts_wav)

        with open(muse_info["out_mp4_path"], "rb") as f:
            mp4_bytes = f.read()

    return {
        "ok": True,
        "mode": "voice_to_video",
        "worker_version": WORKER_VERSION,
        "elapsed_s": round(time.time() - t0, 3),
        "video_b64": base64.b64encode(mp4_bytes).decode("utf-8"),
        "video_mime": "video/mp4",
        "debug": {
            "tts": tts_info,
            "musetalk": {k: v for k, v in muse_info.items() if k != "out_mp4_path"},
            "paths": {
                "MUSE_REPO": MUSE_REPO,
                "MUSE_CONFIG": MUSE_CONFIG,
                "VOICES_DIR": VOICES_DIR,
            }
        }
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode in ("echo", "debug"):
            return mode_echo()

        if mode in ("muse_debug", "musedebug"):
            return mode_muse_debug()

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use echo|muse_debug|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})
