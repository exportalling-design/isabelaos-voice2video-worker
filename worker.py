# /app/worker.py
# RunPod Serverless Worker — IsabelaOS Voice2Video (MuseTalk)
# ✅ Usa el python REAL del container (conda) => sys.executable (/opt/conda/bin/python)
# ✅ No toca /runpod-volume (solo lee)
# ✅ Repo MuseTalk en /runpod-volume/volume_old/MuseTalk (tu ruta real)
# ✅ Modes: scan, echo, voice2video

import os
import sys
import json
import time
import base64
import shutil
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, Tuple

import runpod

# --------------------------------------------------
# ENV
# --------------------------------------------------
RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))     # < 600
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "25"))

# ✅ Python real del container (Conda)
# - En tu caso, sys.executable suele ser /opt/conda/bin/python
PY_CONTAINER = os.environ.get("PY_CONTAINER", sys.executable)

# MuseTalk repo (ruta real que ya te detectó)
MUSE_REPO = os.environ.get("MUSE_REPO", os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"))

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _now() -> float:
    return time.time()

def _tail(s: str, n: int = 1800) -> str:
    return (s or "")[-n:]

def _clean_env(repo_root: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if repo_root:
        env["PYTHONPATH"] = repo_root + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    return env

def _run(cmd, cwd=None, env=None, timeout=HARD_TIMEOUT_SEC) -> Tuple[int, str]:
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    out_lines = []
    start = _now()
    try:
        while True:
            if p.poll() is not None:
                break
            line = p.stdout.readline()
            if line:
                out_lines.append(line)
            if _now() - start > timeout:
                try:
                    p.kill()
                except Exception:
                    pass
                out_lines.append("\n[TIMEOUT] killed process\n")
                return 124, "".join(out_lines)
        rest = p.stdout.read()
        if rest:
            out_lines.append(rest)
        return p.returncode or 0, "".join(out_lines)
    except Exception as ex:
        try:
            p.kill()
        except Exception:
            pass
        out_lines.append(f"\n[EXCEPTION] {ex}\n")
        return 1, "".join(out_lines)

def _looks_like_url(s: Any) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=90) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)

# --------------------------------------------------
# Checks
# --------------------------------------------------
def _repo_check() -> Dict[str, Any]:
    exists = os.path.isdir(MUSE_REPO)
    infer_py = os.path.join(MUSE_REPO, "scripts", "inference.py")
    return {
        "muse_repo": MUSE_REPO,
        "repo_exists": exists,
        "has_inference_py": os.path.isfile(infer_py),
        "inference_py": infer_py,
    }

def _container_import_check() -> Dict[str, Any]:
    # Importa en el python REAL del container (conda)
    env = _clean_env(None)
    cmd = [
        PY_CONTAINER, "-c",
        "import sys; "
        "print('PY=', sys.executable); "
        "import cv2, mmcv, mmengine; "
        "print('OK_cv2_mmcv_mmengine'); "
        "try:\n"
        " import mmpose; print('OK_mmpose')\n"
        "except Exception as e:\n"
        " print('NO_mmpose:', type(e).__name__, str(e))\n"
    ]
    code, out = _run(cmd, env=env, timeout=SCAN_TIMEOUT_SEC)
    return {
        "py_container": PY_CONTAINER,
        "code": code,
        "out_tail": _tail(out),
        "ok": code == 0,
    }

# --------------------------------------------------
# MuseTalk run (NO modifica tu config; asume que tu repo ya está preparado)
# --------------------------------------------------
def _musetalk_infer(input_mp4: str, audio_wav: str) -> Dict[str, Any]:
    if not os.path.isdir(MUSE_REPO):
        raise RuntimeError(f"MuseTalk repo not found: {MUSE_REPO}")
    if not os.path.isfile(os.path.join(MUSE_REPO, "scripts", "inference.py")):
        raise RuntimeError("MuseTalk scripts/inference.py not found in repo")

    env = _clean_env(MUSE_REPO)

    # ⚠️ OJO: MuseTalk a menudo lee rutas desde inference_config.json
    # Este worker NO lo edita (para no romper tu setup). Solo lo ejecuta.
    cmd = [PY_CONTAINER, "-u", "scripts/inference.py", "--inference_config", "inference_config.json"]

    code, out = _run(cmd, cwd=MUSE_REPO, env=env, timeout=HARD_TIMEOUT_SEC)
    if code != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

    # best-effort: buscar mp4 más nuevo en el repo
    newest = None
    newest_mtime = 0.0
    for root, _, files in os.walk(MUSE_REPO):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                fp = os.path.join(root, fn)
                try:
                    mt = os.path.getmtime(fp)
                    if mt > newest_mtime:
                        newest_mtime = mt
                        newest = fp
                except Exception:
                    pass

    return {
        "ok": True,
        "python_used": PY_CONTAINER,
        "output_mp4_guess": newest,
        "log_tail": _tail(out),
    }

# --------------------------------------------------
# Modes
# --------------------------------------------------
def mode_scan() -> Dict[str, Any]:
    return {
        "ok": True,
        "msg": "SCAN_OK",
        "repo": _repo_check(),
        "py_container": PY_CONTAINER,
        "sys_executable": sys.executable,
    }

def mode_echo() -> Dict[str, Any]:
    repo = _repo_check()
    chk = _container_import_check()
    return {
        "ok": True,
        "msg": "ECHO_OK",
        "repo": repo,
        "imports": chk,
        "py_container": PY_CONTAINER,
        "sys_executable": sys.executable,
    }

def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    start = _now()
    repo = _repo_check()
    if not repo["repo_exists"]:
        raise RuntimeError(f"MuseTalk repo not found at {MUSE_REPO}")

    tmp = tempfile.mkdtemp(prefix="v2v_")
    try:
        in_mp4 = os.path.join(tmp, "input.mp4")
        wav = os.path.join(tmp, "audio.wav")

        if _looks_like_url(inp.get("video_url")):
            _download(inp["video_url"], in_mp4)
        elif isinstance(inp.get("video_b64"), str) and inp["video_b64"]:
            with open(in_mp4, "wb") as f:
                f.write(base64.b64decode(inp["video_b64"]))
        else:
            raise RuntimeError("Missing video_url or video_b64")

        if _looks_like_url(inp.get("audio_url")):
            _download(inp["audio_url"], wav)
        elif isinstance(inp.get("audio_b64"), str) and inp["audio_b64"]:
            with open(wav, "wb") as f:
                f.write(base64.b64decode(inp["audio_b64"]))
        else:
            raise RuntimeError("Missing audio_url or audio_b64")

        info = _musetalk_infer(in_mp4, wav)
        elapsed = int((_now() - start) * 1000)

        return {
            "ok": True,
            "msg": "VOICE2VIDEO_OK",
            "execution_ms": elapsed,
            "repo": repo,
            "python_used": info.get("python_used"),
            "output_mp4_guess": info.get("output_mp4_guess"),
            "log_tail": info.get("log_tail"),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------
# Handler
# --------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = event.get("input") if isinstance(event, dict) else None
        if not isinstance(inp, dict):
            return {"ok": False, "error": "Missing or invalid input (expected JSON with field 'input')"}

        mode = str(inp.get("mode", "scan")).strip().lower()

        if mode == "scan":
            return mode_scan()
        if mode == "echo":
            return mode_echo()
        if mode in ("voice2video", "v2v"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
