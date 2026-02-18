# /app/worker.py
# RunPod Serverless Worker - IsabelaOS voice2video (XTTS -> MuseTalk)
# Modes:
#  - {"input": {"mode":"echo"}}
#  - {"input": {"mode":"voice2video", "video_url": "...", "text":"...", "lang":"es", "voice":"female"}}
#
# Notes:
#  - MuseTalk is executed using venv python from volume: /runpod-volume/musetalk_ok/bin/python
#  - MuseTalk repo expected in volume: /runpod-volume/volume_old/MuseTalk
#  - inference_config.json expected in repo root (we pass absolute path)

import os
import re
import gc
import json
import time
import base64
import shutil
import tempfile
import traceback
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import runpod


# -----------------------------
# Paths (FIXED to your volume)
# -----------------------------
RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume").strip() or "/runpod-volume"

VOICES_DIR = os.path.join(RUNPOD_VOLUME_PATH, "voices")
FEMALE_REF_WAV = os.path.join(VOICES_DIR, "female_ref.wav")
MALE_REF_WAV = os.path.join(VOICES_DIR, "male_ref.wav")

# MuseTalk repo + venv python (from your echo results)
MUSE_REPO_PICKED = os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk")
MUSE_CONFIG_PICKED = os.path.join(MUSE_REPO_PICKED, "inference_config.json")
MUSE_PYTHON = os.path.join(RUNPOD_VOLUME_PATH, "musetalk_ok", "bin", "python")

# Local scripts inside container
TTS_SCRIPT = "/app/tts_generate.py"


# -----------------------------
# Helpers
# -----------------------------
def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def _require(p: str, label: str) -> None:
    if not _exists(p):
        raise RuntimeError(f"Missing {label}: {p}")


def _clean_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Create a safe env for subprocesses (keep minimal + volume vars)."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/root"),
        "RUNPOD_VOLUME_PATH": RUNPOD_VOLUME_PATH,
        "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED", "1"),
        "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        "TTS_USE_GPU": os.environ.get("TTS_USE_GPU", "1"),
    }
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def _run(cmd, cwd=None, env=None, timeout=None) -> Tuple[int, str]:
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    out_lines = []
    start = time.time()
    while True:
        line = p.stdout.readline()
        if line:
            out_lines.append(line)
        if p.poll() is not None:
            # drain remaining
            rest = p.stdout.read()
            if rest:
                out_lines.append(rest)
            break
        if timeout and (time.time() - start) > timeout:
            try:
                p.kill()
            except Exception:
                pass
            out_lines.append("\n[TIMEOUT_KILLED]\n")
            break
    out = "".join(out_lines)
    return p.returncode or 0, out


def _download_to(url: str, out_path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)


def _find_newest_mp4(search_dir: str) -> Optional[str]:
    p = Path(search_dir)
    if not p.exists():
        return None
    mp4s = list(p.rglob("*.mp4"))
    if not mp4s:
        return None
    mp4s.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(mp4s[0])


def _tail(s: str, n: int = 2200) -> str:
    if not s:
        return ""
    return s[-n:]


def _looks_like_cuda_ecc(out: str) -> bool:
    if not out:
        return False
    o = out.lower()
    return ("uncorrectable ecc" in o) or ("cuda error" in o)


# -----------------------------
# Modes
# -----------------------------
def mode_echo() -> Dict[str, Any]:
    checks = {
        "voices_dir_exists": _exists(VOICES_DIR),
        "female_ref_exists": _exists(FEMALE_REF_WAV),
        "male_ref_exists": _exists(MALE_REF_WAV),
        "muse_repo_exists": _exists(MUSE_REPO_PICKED),
        "muse_scripts_inference_exists": _exists(os.path.join(MUSE_REPO_PICKED, "scripts", "inference.py")),
        "muse_config_exists": _exists(MUSE_CONFIG_PICKED),
        "muse_venv_python_exists": _exists(MUSE_PYTHON),
    }
    return {
        "ok": True,
        "msg": "ECHO_OK",
        "base": RUNPOD_VOLUME_PATH,
        "python": shutil.which("python3") or "/usr/bin/python3",
        "env": {
            "RUNPOD_VOLUME_PATH": RUNPOD_VOLUME_PATH,
            "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED", ""),
            "TTS_USE_GPU": os.environ.get("TTS_USE_GPU", ""),
        },
        "checks": checks,
        "paths": {
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "MUSE_REPO_PICKED": MUSE_REPO_PICKED,
            "MUSE_CONFIG_PICKED": MUSE_CONFIG_PICKED,
            "MUSE_PYTHON": MUSE_PYTHON,
        },
    }


def _tts_make_wav(text: str, lang: str, speaker_wav: str, out_wav: str) -> Dict[str, Any]:
    """
    Calls /app/tts_generate.py.
    First tries GPU (if enabled). If ECC/CUDA error, retries CPU automatically.
    """
    _require(TTS_SCRIPT, "TTS script")
    _require(speaker_wav, "speaker_wav")

    base_cmd = [
        "/usr/local/bin/python3",
        "-u",
        TTS_SCRIPT,
        "--text",
        text,
        "--lang",
        lang,
        "--speaker_wav",
        speaker_wav,
        "--out_wav",
        out_wav,
    ]

    # Attempt 1: current env (likely GPU)
    code, out = _run(base_cmd, env=_clean_env(), timeout=600)
    if code == 0 and _exists(out_wav):
        return {"ok": True, "device": "gpu_or_default", "log_tail": _tail(out)}

    # If CUDA/ECC -> retry CPU
    if _looks_like_cuda_ecc(out):
        cpu_env = _clean_env(
            {
                "TTS_USE_GPU": "0",
                "CUDA_VISIBLE_DEVICES": "",  # force CPU
            }
        )
        code2, out2 = _run(base_cmd, env=cpu_env, timeout=900)
        if code2 == 0 and _exists(out_wav):
            return {"ok": True, "device": "cpu_fallback", "log_tail": _tail(out2)}

        raise RuntimeError("TTS failed (CPU fallback also failed)\n" + _tail(out2))

    raise RuntimeError("TTS failed\n" + _tail(out))


def _musetalk_infer(repo_root: str, config_path: str, input_mp4: str, audio_wav: str) -> Dict[str, Any]:
    """
    Runs MuseTalk inference using venv python from volume.
    IMPORTANT: use MUSE_PYTHON, not system python.
    """
    _require(repo_root, "MuseTalk repo root")
    _require(os.path.join(repo_root, "scripts", "inference.py"), "MuseTalk scripts/inference.py")
    _require(config_path, "MuseTalk inference_config.json")
    _require(MUSE_PYTHON, "MuseTalk venv python")
    _require(input_mp4, "input video")
    _require(audio_wav, "audio wav")

    # MuseTalk expects files inside repo_root/inputs typically. We'll place there.
    inputs_dir = os.path.join(repo_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    v_path = os.path.join(inputs_dir, "input.mp4")
    a_path = os.path.join(inputs_dir, "audio.wav")
    shutil.copy2(input_mp4, v_path)
    shutil.copy2(audio_wav, a_path)

    # Make sure repo-root python can import local "musetalk" package from repo
    env = _clean_env(
        {
            "PYTHONPATH": repo_root,
        }
    )

    cmd = [
        MUSE_PYTHON,
        "-u",
        "scripts/inference.py",
        "--inference_config",
        config_path,          # absolute
        "--bbox_shift",
        "0",
        "--use_float16",
    ]

    code, out = _run(cmd, cwd=repo_root, env=env, timeout=1800)
    if code != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

    # Find newest MP4 in repo results
    out_mp4 = _find_newest_mp4(os.path.join(repo_root, "results")) or _find_newest_mp4(repo_root)
    if not out_mp4:
        raise RuntimeError("MuseTalk finished but no MP4 found in results/")

    return {"ok": True, "out_mp4": out_mp4, "log_tail": _tail(out)}


def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    video_url = (inp.get("video_url") or "").strip()
    text = (inp.get("text") or "").strip()
    lang = (inp.get("lang") or "es").strip()
    voice = (inp.get("voice") or "female").strip().lower()
    return_b64 = bool(inp.get("return_b64", True))

    if not video_url:
        raise RuntimeError("Missing video_url")
    if not text:
        raise RuntimeError("Missing text")

    speaker_wav = FEMALE_REF_WAV if voice != "male" else MALE_REF_WAV

    _require(VOICES_DIR, "voices dir")
    _require(speaker_wav, "speaker reference wav")
    _require(MUSE_REPO_PICKED, "MuseTalk repo")
    _require(MUSE_CONFIG_PICKED, "MuseTalk config")
    _require(MUSE_PYTHON, "MuseTalk venv python")

    # temp workspace
    work = tempfile.mkdtemp(prefix="voice2video_")
    in_mp4 = os.path.join(work, "in.mp4")
    tts_wav = os.path.join(work, "tts.wav")

    # download video
    _download_to(video_url, in_mp4)

    # tts
    tts_info = _tts_make_wav(text=text, lang=lang, speaker_wav=speaker_wav, out_wav=tts_wav)

    # musetalk
    musetalk_info = _musetalk_infer(
        repo_root=MUSE_REPO_PICKED,
        config_path=MUSE_CONFIG_PICKED,
        input_mp4=in_mp4,
        audio_wav=tts_wav,
    )

    out_mp4 = musetalk_info["out_mp4"]

    # Return
    resp = {
        "ok": True,
        "tts": tts_info,
        "musetalk": {k: v for k, v in musetalk_info.items() if k != "out_mp4"},
        "out_mp4_path": out_mp4,
    }

    if return_b64:
        # careful: large outputs; ok for short clips
        with open(out_mp4, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        resp["out_mp4_b64"] = b64

    # cleanup a bit
    try:
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass
    gc.collect()

    return resp


# -----------------------------
# RunPod handler
# -----------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # RunPod sends {"id":..., "input": {...}}
        inp = job.get("input") if isinstance(job, dict) else None

        # Some clients may send raw input without wrapper
        if inp is None and isinstance(job, dict) and ("mode" in job):
            inp = job

        if not isinstance(inp, dict):
            return {"ok": False, "error": "Job has missing field(s): input (expected {'input':{...}})"}

        mode = (inp.get("mode") or "voice2video").strip().lower()

        if mode == "echo":
            return mode_echo()

        if mode in ("voice2video", "voice_to_video"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})
