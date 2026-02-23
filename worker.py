# /app/worker.py
# RunPod Serverless Worker — IsabelaOS Voice2Video (XTTS -> MuseTalk)
# ✅ No reinstala nada
# ✅ Usa SOLO lo que ya existe en el volumen
# ✅ Prioriza musetalk_ok
# ✅ No depende de import musetalk
# ✅ Modes: scan, echo, voice2video

import os
import json
import time
import base64
import shutil
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, Tuple, List

import runpod

# --------------------------------------------------
# ENV
# --------------------------------------------------

RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
COQUI_TOS_AGREED = "1"
HARD_TIMEOUT_SEC = 560
SCAN_TIMEOUT_SEC = 20

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _now():
    return time.time()

def _tail(s: str, n: int = 1200):
    return (s or "")[-n:]

def _run(cmd, cwd=None, env=None, timeout=HARD_TIMEOUT_SEC):
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    start = _now()
    out = []
    while True:
        if p.poll() is not None:
            break
        if _now() - start > timeout:
            p.kill()
            return 124, "".join(out)
        line = p.stdout.readline()
        if line:
            out.append(line)
    out.append(p.stdout.read() or "")
    return p.returncode or 0, "".join(out)

def _clean_env(repo_root=None):
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["COQUI_TOS_AGREED"] = "1"

    if repo_root:
        env["PYTHONPATH"] = repo_root + (
            ":" + env.get("PYTHONPATH", "")
            if env.get("PYTHONPATH") else ""
        )
    return env

# --------------------------------------------------
# MuseTalk Repo detection
# --------------------------------------------------

def _pick_musetalk_repo():
    candidates = [
        os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"),
        os.path.join(RUNPOD_VOLUME_PATH, "MuseTalk"),
    ]
    for p in candidates:
        if os.path.isfile(os.path.join(p, "scripts", "inference.py")):
            return p
    return None

# --------------------------------------------------
# Python detection
# --------------------------------------------------

def _list_candidate_pythons():
    candidates = []

    # 🔥 priorizar musetalk_ok
    preferred = [
        os.path.join(RUNPOD_VOLUME_PATH, "musetalk_ok", "bin", "python"),
        os.path.join(RUNPOD_VOLUME_PATH, "musetalk_ok_persist", "bin", "python"),
    ]

    for p in preferred:
        if os.path.isfile(p):
            candidates.append(p)

    # buscar otros venvs
    for root, dirs, files in os.walk(RUNPOD_VOLUME_PATH):
        if root.endswith("/bin"):
            py = os.path.join(root, "python")
            if os.path.isfile(py):
                candidates.append(py)

    sys_py = shutil.which("python3")
    if sys_py:
        candidates.append(sys_py)

    # eliminar duplicados
    seen = set()
    final = []
    for p in candidates:
        if p not in seen:
            final.append(p)
            seen.add(p)

    return final

def _probe_python(py, repo_root):
    env = _clean_env(repo_root)

    code, out = _run(
        [py, "-c", "import cv2, mmcv, mmengine, mmpose; print('OK')"],
        cwd=repo_root,
        env=env,
        timeout=SCAN_TIMEOUT_SEC,
    )

    return {
        "py": py,
        "ok": code == 0,
        "tail": _tail(out),
    }

def _select_best_python(repo_root):
    candidates = _list_candidate_pythons()
    results = []

    for py in candidates:
        r = _probe_python(py, repo_root)
        results.append(r)

    # el primero que tenga openmmlab OK gana
    for r in results:
        if r["ok"]:
            return r["py"], r

    # fallback
    return candidates[-1], results[-1]

# --------------------------------------------------
# MuseTalk Inference
# --------------------------------------------------

def _musetalk_infer(repo_root, input_mp4, audio_wav):
    py, probe = _select_best_python(repo_root)

    env = _clean_env(repo_root)

    cmd = [
        py,
        "-u",
        "scripts/inference.py",
        "--inference_config",
        "inference_config.json",
    ]

    code, out = _run(cmd, cwd=repo_root, env=env)

    if code != 0:
        raise RuntimeError("MuseTalk failed\n" + _tail(out))

    # buscar mp4 más nuevo
    newest = None
    newest_mtime = 0
    for root, _, files in os.walk(repo_root):
        for f in files:
            if f.endswith(".mp4"):
                fp = os.path.join(root, f)
                mt = os.path.getmtime(fp)
                if mt > newest_mtime:
                    newest = fp
                    newest_mtime = mt

    return {
        "python_used": py,
        "probe": probe,
        "output_mp4_guess": newest,
        "log_tail": _tail(out),
    }

# --------------------------------------------------
# Modes
# --------------------------------------------------

def mode_scan():
    repo_root = _pick_musetalk_repo()
    py, probe = _select_best_python(repo_root)
    return {
        "ok": True,
        "repo_root": repo_root,
        "picked_python": py,
        "probe": probe,
    }

def mode_echo():
    repo_root = _pick_musetalk_repo()
    py, probe = _select_best_python(repo_root)

    return {
        "ok": True,
        "base": RUNPOD_VOLUME_PATH,
        "repo_root": repo_root,
        "picked_python": py,
        "probe": probe,
    }

def mode_voice2video(inp):
    repo_root = _pick_musetalk_repo()
    if not repo_root:
        raise RuntimeError("MuseTalk repo not found")

    tmp = tempfile.mkdtemp(prefix="v2v_")
    try:
        in_mp4 = os.path.join(tmp, "input.mp4")
        tts_wav = os.path.join(tmp, "audio.wav")

        if "video_url" in inp:
            urllib.request.urlretrieve(inp["video_url"], in_mp4)
        else:
            raise RuntimeError("Missing video_url")

        if "audio_url" in inp:
            urllib.request.urlretrieve(inp["audio_url"], tts_wav)
        else:
            raise RuntimeError("Missing audio_url")

        info = _musetalk_infer(repo_root, in_mp4, tts_wav)

        return {
            "ok": True,
            "repo_root": repo_root,
            "python_used": info["python_used"],
            "output_mp4_guess": info["output_mp4_guess"],
            "log_tail": info["log_tail"],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------
# Handler
# --------------------------------------------------

def handler(event):
    try:
        inp = event.get("input", {})
        mode = inp.get("mode", "voice2video").lower()

        if mode == "scan":
            return mode_scan()
        if mode == "echo":
            return mode_echo()
        if mode in ("voice2video", "v2v"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }

runpod.serverless.start({"handler": handler})
