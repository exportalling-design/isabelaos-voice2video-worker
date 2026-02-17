import os
import re
import io
import json
import time
import uuid
import shutil
import base64
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, Tuple

import runpod


# ----------------------------
# Paths (hard defaults)
# ----------------------------
VOLUME_BASE = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")

VOICES_DIR = os.path.join(VOLUME_BASE, "voices")
FEMALE_REF_WAV = os.path.join(VOICES_DIR, "female_ref.wav")
MALE_REF_WAV = os.path.join(VOICES_DIR, "male_ref.wav")

# MuseTalk (lo que YA tenés)
MUSE_VENV_PY = os.environ.get("MUSE_PYTHON", os.path.join(VOLUME_BASE, "musetalk_ok", "bin", "python"))

# Repo viejo que sí tiene scripts/inference.py + musetalk/
DEFAULT_MUSE_REPO = os.environ.get("MUSE_REPO", os.path.join(VOLUME_BASE, "volume_old", "MuseTalk"))

# Config correcto (SIN el punto raro)
DEFAULT_MUSE_CONFIG = os.environ.get("MUSE_CONFIG", os.path.join(DEFAULT_MUSE_REPO, "inference_config.json"))

# TTS script
TTS_SCRIPT = "/app/tts_generate.py"


# ----------------------------
# Helpers
# ----------------------------
def _require_file(path: str, label: str):
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _require_dir(path: str, label: str):
    if not os.path.isdir(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _run(cmd, cwd=None, env=None, stdin_text=None, timeout=None):
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = p.communicate(stdin_text, timeout=timeout)
    if p.returncode != 0:
        tail = "\n".join(out.splitlines()[-80:])
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}\n")
    return out

def _download_to(url: str, out_path: str):
    # Descarga directa (Supabase public sirve)
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)

def _clean_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    # env limpio pero conserva lo básico
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if extra:
        env.update(extra)
    return env


# ----------------------------
# MuseTalk repo detection
# ----------------------------
def _scan_musetalk() -> Dict[str, Any]:
    """
    Encuentra:
      - repo con scripts/inference.py y carpeta musetalk/
      - config inference_config.json (sin el punto final raro)
    """
    candidates = [
        DEFAULT_MUSE_REPO,
        os.path.join(VOLUME_BASE, "MuseTalk"),
        os.path.join(VOLUME_BASE, "volume_old", "MuseTalk"),
    ]

    picked_repo = None
    for c in candidates:
        if os.path.isfile(os.path.join(c, "scripts", "inference.py")) and os.path.isdir(os.path.join(c, "musetalk")):
            picked_repo = c
            break

    # config
    picked_cfg = None
    if picked_repo:
        cfg1 = os.path.join(picked_repo, "inference_config.json")
        if os.path.isfile(cfg1):
            picked_cfg = cfg1

    return {
        "repo_root": picked_repo,
        "config": picked_cfg,
        "venv_python": MUSE_VENV_PY,
    }


# ----------------------------
# TTS
# ----------------------------
def _tts_make_wav(text: str, lang: str, voice: str, out_wav: str):
    _require_dir(VOICES_DIR, "VOICES_DIR")
    speaker = FEMALE_REF_WAV if voice.lower() == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")

    cmd = [
        "/usr/local/bin/python3",
        "-u",
        TTS_SCRIPT,
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav,
    ]

    # auto-accept si algo pide input
    _run(cmd, env=_clean_env(), stdin_text="y\n")


# ----------------------------
# MuseTalk inference (usando venv del volumen)
# ----------------------------
def _musetalk_infer(repo_root: str, config_json: str, input_mp4: str, audio_wav: str, out_dir: str) -> str:
    _require_dir(repo_root, "MuseTalk repo_root")
    _require_file(os.path.join(repo_root, "scripts", "inference.py"), "scripts/inference.py")
    _require_dir(os.path.join(repo_root, "musetalk"), "musetalk package folder")

    _require_file(config_json, "inference_config.json")
    _require_file(input_mp4, "input_mp4")
    _require_file(audio_wav, "audio_wav")
    _require_file(MUSE_VENV_PY, "MUSE_PYTHON (venv python)")

    # MuseTalk suele usar archivos dentro de su repo (inputs / results).
    # Para no pelear, lo corremos desde el repo y copiamos el input a su carpeta inputs.
    inputs_dir = os.path.join(repo_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    local_mp4 = os.path.join(inputs_dir, "input.mp4")
    local_wav = os.path.join(inputs_dir, "audio.wav")
    shutil.copyfile(input_mp4, local_mp4)
    shutil.copyfile(audio_wav, local_wav)

    # Algunas versiones usan args por default leyendo inputs.
    # Tu error anterior era: "No module named musetalk" / "No module named diffusers/mmpose".
    # Eso se arregla así:
    #  - usar el python del venv (que tiene deps)
    #  - exportar PYTHONPATH apuntando al repo para que "import musetalk" funcione
    env = _clean_env({
        "PYTHONPATH": repo_root,
    })

    cmd = [
        MUSE_VENV_PY,
        "-u",
        "scripts/inference.py",
        "--inference_config", os.path.basename(config_json),
        "--bbox_shift", "0",
        "--use_float16",
    ]

    # Asegurar que inference_config.json esté en el cwd con el nombre esperado
    # (MuseTalk normalmente espera "inference_config.json" en el repo root)
    if os.path.basename(config_json) != "inference_config.json":
        # pero por si te pasan otro nombre
        pass

    # Copiamos config al repo root como inference_config.json
    cfg_target = os.path.join(repo_root, "inference_config.json")
    if os.path.abspath(config_json) != os.path.abspath(cfg_target):
        shutil.copyfile(config_json, cfg_target)

    _run(cmd, cwd=repo_root, env=env)

    # Buscar mp4 de salida dentro del repo (varía por forks).
    # Vamos a buscar el mp4 más nuevo en repo_root
    newest = None
    newest_t = -1.0
    for root, _, files in os.walk(repo_root):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                p = os.path.join(root, fn)
                try:
                    t = os.path.getmtime(p)
                    if t > newest_t:
                        newest_t = t
                        newest = p
                except:
                    pass

    if not newest:
        raise RuntimeError("MuseTalk finished but no output mp4 found in repo")

    out_mp4 = os.path.join(out_dir, "out.mp4")
    shutil.copyfile(newest, out_mp4)
    return out_mp4


# ----------------------------
# Main modes
# ----------------------------
def _mode_echo() -> Dict[str, Any]:
    scan = _scan_musetalk()
    checks = {
        "voices_dir_exists": os.path.isdir(VOICES_DIR),
        "female_ref_exists": os.path.isfile(FEMALE_REF_WAV),
        "male_ref_exists": os.path.isfile(MALE_REF_WAV),
        "muse_repo_exists": bool(scan["repo_root"]) and os.path.isdir(scan["repo_root"]),
        "muse_scripts_inference_exists": bool(scan["repo_root"]) and os.path.isfile(os.path.join(scan["repo_root"], "scripts", "inference.py")),
        "muse_config_exists": bool(scan["config"]) and os.path.isfile(scan["config"]),
        "muse_venv_python_exists": os.path.isfile(scan["venv_python"]),
    }
    return {
        "ok": True,
        "msg": "ECHO_OK",
        "base": VOLUME_BASE,
        "env": {
            "RUNPOD_VOLUME_PATH": os.environ.get("RUNPOD_VOLUME_PATH"),
            "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
            "TTS_USE_GPU": os.environ.get("TTS_USE_GPU"),
        },
        "paths": {
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "MUSE_REPO_PICKED": scan["repo_root"],
            "MUSE_CONFIG_PICKED": scan["config"],
            "MUSE_PYTHON": scan["venv_python"],
        },
        "checks": checks,
        "python": "/usr/local/bin/python3",
    }

def _mode_scan() -> Dict[str, Any]:
    scan = _scan_musetalk()
    return {
        "ok": True,
        "msg": "SCAN_OK",
        "scan": scan,
    }

def voice_to_video(inp: Dict[str, Any]) -> Dict[str, Any]:
    text = inp.get("text", "Hola, es una prueba.")
    lang = inp.get("lang", "es")
    voice = inp.get("voice", "female")  # female | male
    video_url = inp.get("video_url")
    if not video_url:
        raise RuntimeError("Missing video_url")

    scan = _scan_musetalk()
    repo_root = scan["repo_root"]
    config_json = scan["config"] or DEFAULT_MUSE_CONFIG

    if not repo_root:
        raise RuntimeError("MuseTalk repo not found in volume. Expected /runpod-volume/volume_old/MuseTalk (or set MUSE_REPO).")
    if not os.path.isfile(config_json):
        raise RuntimeError(f"MuseTalk config not found: {config_json}")
    _require_file(MUSE_VENV_PY, "MUSE_PYTHON (venv python)")

    with tempfile.TemporaryDirectory() as td:
        in_mp4 = os.path.join(td, "input.mp4")
        tts_wav = os.path.join(td, "tts.wav")
        out_dir = os.path.join(td, "out")
        os.makedirs(out_dir, exist_ok=True)

        _download_to(video_url, in_mp4)
        _tts_make_wav(text=text, lang=lang, voice=voice, out_wav=tts_wav)

        out_mp4_path = _musetalk_infer(
            repo_root=repo_root,
            config_json=config_json,
            input_mp4=in_mp4,
            audio_wav=tts_wav,
            out_dir=out_dir,
        )

        with open(out_mp4_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "ok": True,
            "video_b64": b64,
            "meta": {
                "voice": voice,
                "lang": lang,
                "repo_root": repo_root,
                "config": config_json,
            }
        }


def handler(event):
    try:
        inp = event.get("input", {}) or {}
        mode = (inp.get("mode") or "voice2video").lower()

        if mode == "echo":
            return _mode_echo()

        if mode == "scan":
            return _mode_scan()

        if mode == "voice2video":
            return voice_to_video(inp)

        raise RuntimeError(f"Unknown mode: {mode}")

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
