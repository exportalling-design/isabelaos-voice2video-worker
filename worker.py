# /app/worker.py
# RunPod Serverless Worker - IsabelaOS voice2video (XTTS -> MuseTalk)
# Modes:
#  - {"input": {"mode":"echo"}}
#  - {"input": {"mode":"muse_debug"}}
#  - {"input": {"mode":"voice2video", "video_url": "...", "text":"...", "lang":"es", "voice":"female"}}

import os
import gc
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

RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume").strip() or "/runpod-volume"

VOICES_DIR = os.path.join(RUNPOD_VOLUME_PATH, "voices")
FEMALE_REF_WAV = os.path.join(VOICES_DIR, "female_ref.wav")
MALE_REF_WAV = os.path.join(VOICES_DIR, "male_ref.wav")

# Your known good locations:
MUSE_REPO_PICKED = os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk")
MUSE_CONFIG_PICKED = os.path.join(MUSE_REPO_PICKED, "inference_config.json")
MUSE_PYTHON = os.path.join(RUNPOD_VOLUME_PATH, "musetalk_ok", "bin", "python")

TTS_SCRIPT = "/app/tts_generate.py"


def _exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def _require(p: str, label: str) -> None:
    if not _exists(p):
        raise RuntimeError(f"Missing {label}: {p}")


def _clean_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/root"),
        "RUNPOD_VOLUME_PATH": RUNPOD_VOLUME_PATH,
        "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED", "1"),
        "HF_HUB_ENABLE_HF_TRANSFER": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        "TTS_USE_GPU": os.environ.get("TTS_USE_GPU", "1"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONUNBUFFERED": "1",
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
    return (p.returncode or 0), out


def _tail(s: str, n: int = 2600) -> str:
    return s[-n:] if s else ""


def _download_to(url: str, out_path: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(out_path, "wb") as f:
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


def _looks_like_cuda_ecc(out: str) -> bool:
    if not out:
        return False
    o = out.lower()
    return ("uncorrectable ecc" in o) or ("cuda error" in o)


# -----------------------------
# MuseTalk deps auto-fix (cv2)
# -----------------------------
def _ensure_musetalk_deps() -> Dict[str, Any]:
    """
    Guarantees cv2 is importable inside the MuseTalk venv.
    Steps:
      1) verify pip exists in venv
      2) if not, run ensurepip
      3) install opencv-python-headless
      4) verify import cv2
    """
    _require(MUSE_PYTHON, "MuseTalk venv python")

    env = _clean_env()

    info = {"ok": True, "steps": {}, "python": MUSE_PYTHON}

    # Show python identity
    c0, o0 = _run([MUSE_PYTHON, "-c", "import sys; print(sys.executable); print(sys.version)"], env=env, timeout=60)
    info["steps"]["python_id_ok"] = (c0 == 0)
    info["steps"]["python_id_tail"] = _tail(o0)

    # Check if cv2 already works
    c1, o1 = _run([MUSE_PYTHON, "-c", "import cv2; print('cv2_ok', cv2.__version__)"], env=env, timeout=120)
    if c1 == 0:
        info["steps"]["cv2_present"] = True
        info["steps"]["cv2_check_tail"] = _tail(o1)
        return info

    info["steps"]["cv2_present"] = False
    info["steps"]["cv2_check_tail"] = _tail(o1)

    # Check pip
    c2, o2 = _run([MUSE_PYTHON, "-m", "pip", "--version"], env=env, timeout=120)
    info["steps"]["pip_ok_before"] = (c2 == 0)
    info["steps"]["pip_before_tail"] = _tail(o2)

    if c2 != 0:
        # ensurepip
        c3, o3 = _run([MUSE_PYTHON, "-m", "ensurepip", "--upgrade"], env=env, timeout=300)
        info["steps"]["ensurepip_ok"] = (c3 == 0)
        info["steps"]["ensurepip_tail"] = _tail(o3)
        if c3 != 0:
            raise RuntimeError("ensurepip failed in MuseTalk venv\n" + _tail(o3))

        # re-check pip
        c4, o4 = _run([MUSE_PYTHON, "-m", "pip", "--version"], env=env, timeout=120)
        info["steps"]["pip_ok_after"] = (c4 == 0)
        info["steps"]["pip_after_tail"] = _tail(o4)
        if c4 != 0:
            raise RuntimeError("pip still missing after ensurepip\n" + _tail(o4))

    # Install OpenCV (headless)
    install_cmd = [
        MUSE_PYTHON, "-m", "pip", "install", "--no-cache-dir",
        "opencv-python-headless"
    ]
    c5, o5 = _run(install_cmd, env=env, timeout=1200)
    info["steps"]["opencv_install_ok"] = (c5 == 0)
    info["steps"]["opencv_install_tail"] = _tail(o5)
    if c5 != 0:
        raise RuntimeError("opencv install failed in MuseTalk venv\n" + _tail(o5))

    # Verify cv2 import after install
    c6, o6 = _run([MUSE_PYTHON, "-c", "import cv2; print('cv2_ok', cv2.__version__)"], env=env, timeout=120)
    info["steps"]["cv2_ok_after_install"] = (c6 == 0)
    info["steps"]["cv2_after_tail"] = _tail(o6)
    if c6 != 0:
        raise RuntimeError("cv2 still failing after install\n" + _tail(o6))

    return info


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


def mode_muse_debug() -> Dict[str, Any]:
    _require(MUSE_PYTHON, "MuseTalk venv python")
    _require(MUSE_REPO_PICKED, "MuseTalk repo")
    _require(MUSE_CONFIG_PICKED, "MuseTalk config")

    deps = _ensure_musetalk_deps()

    # show pip list line for opencv
    env = _clean_env()
    c, o = _run([MUSE_PYTHON, "-m", "pip", "show", "opencv-python-headless"], env=env, timeout=120)
    return {
        "ok": True,
        "msg": "MUSE_DEBUG_OK",
        "deps": deps,
        "opencv_show_ok": (c == 0),
        "opencv_show_tail": _tail(o),
    }


def _tts_make_wav(text: str, lang: str, speaker_wav: str, out_wav: str) -> Dict[str, Any]:
    _require(TTS_SCRIPT, "TTS script")
    _require(speaker_wav, "speaker_wav")

    cmd = [
        "/usr/local/bin/python3", "-u", TTS_SCRIPT,
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker_wav,
        "--out_wav", out_wav,
    ]

    code, out = _run(cmd, env=_clean_env(), timeout=600)
    if code == 0 and _exists(out_wav):
        return {"ok": True, "device": "gpu_or_default", "log_tail": _tail(out)}

    if _looks_like_cuda_ecc(out):
        cpu_env = _clean_env({"TTS_USE_GPU": "0", "CUDA_VISIBLE_DEVICES": ""})
        code2, out2 = _run(cmd, env=cpu_env, timeout=900)
        if code2 == 0 and _exists(out_wav):
            return {"ok": True, "device": "cpu_fallback", "log_tail": _tail(out2)}
        raise RuntimeError("TTS failed (CPU fallback also failed)\n" + _tail(out2))

    raise RuntimeError("TTS failed\n" + _tail(out))


def _musetalk_infer(repo_root: str, config_path: str, input_mp4: str, audio_wav: str) -> Dict[str, Any]:
    _require(repo_root, "MuseTalk repo root")
    _require(os.path.join(repo_root, "scripts", "inference.py"), "MuseTalk scripts/inference.py")
    _require(config_path, "MuseTalk inference_config.json")
    _require(MUSE_PYTHON, "MuseTalk venv python")
    _require(input_mp4, "input video")
    _require(audio_wav, "audio wav")

    # ✅ hard guarantee cv2 exists in venv
    deps_info = _ensure_musetalk_deps()

    inputs_dir = os.path.join(repo_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    v_path = os.path.join(inputs_dir, "input.mp4")
    a_path = os.path.join(inputs_dir, "audio.wav")
    shutil.copy2(input_mp4, v_path)
    shutil.copy2(audio_wav, a_path)

    env = _clean_env({"PYTHONPATH": repo_root})

    # extra sanity check right before running inference:
    cchk, ochk = _run([MUSE_PYTHON, "-c", "import cv2; print('cv2_ok', cv2.__version__)"], env=env, timeout=120)
    if cchk != 0:
        raise RuntimeError("cv2 still missing right before inference\n" + _tail(ochk))

    cmd = [
        MUSE_PYTHON, "-u",
        "scripts/inference.py",
        "--inference_config", config_path,
        "--bbox_shift", "0",
        "--use_float16",
    ]

    code, out = _run(cmd, cwd=repo_root, env=env, timeout=1800)
    if code != 0:
        raise RuntimeError("MuseTalk inference failed\nCMD: " + " ".join(cmd) + "\n" + _tail(out))

    out_mp4 = _find_newest_mp4(os.path.join(repo_root, "results")) or _find_newest_mp4(repo_root)
    if not out_mp4:
        raise RuntimeError("MuseTalk finished but no MP4 found in results/")

    return {"ok": True, "out_mp4": out_mp4, "deps": deps_info, "log_tail": _tail(out)}


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

    work = tempfile.mkdtemp(prefix="voice2video_")
    in_mp4 = os.path.join(work, "in.mp4")
    tts_wav = os.path.join(work, "tts.wav")

    _download_to(video_url, in_mp4)

    tts_info = _tts_make_wav(text=text, lang=lang, speaker_wav=speaker_wav, out_wav=tts_wav)

    musetalk_info = _musetalk_infer(
        repo_root=MUSE_REPO_PICKED,
        config_path=MUSE_CONFIG_PICKED,
        input_mp4=in_mp4,
        audio_wav=tts_wav,
    )

    out_mp4 = musetalk_info["out_mp4"]

    resp = {
        "ok": True,
        "tts": tts_info,
        "musetalk": {k: v for k, v in musetalk_info.items() if k != "out_mp4"},
        "out_mp4_path": out_mp4,
    }

    if return_b64:
        with open(out_mp4, "rb") as f:
            resp["out_mp4_b64"] = base64.b64encode(f.read()).decode("utf-8")

    try:
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass

    gc.collect()
    return resp


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") if isinstance(job, dict) else None
        if inp is None and isinstance(job, dict) and ("mode" in job):
            inp = job

        if not isinstance(inp, dict):
            return {"ok": False, "error": "Job has missing field(s): input (expected {'input':{...}})"}

        mode = (inp.get("mode") or "voice2video").strip().lower()

        if mode == "echo":
            return mode_echo()
        if mode == "muse_debug":
            return mode_muse_debug()
        if mode in ("voice2video", "voice_to_video"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
