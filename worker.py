# /app/worker.py
# RunPod Serverless Worker — IsabelaOS Voice2Video (XTTS -> MuseTalk)
# ✅ No toca /runpod-volume
# ✅ FIX real: el endpoint usa python del container; instalamos mmcv/mmengine/mmpose EN EL CONTAINER (Dockerfile)
# ✅ Worker fuerza ese python y solo usa el repo MuseTalk desde /runpod-volume
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
from typing import Any, Dict, Optional, Tuple

import runpod

# --------------------------------------------------
# ENV
# --------------------------------------------------
RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
COQUI_TOS_AGREED = os.environ.get("COQUI_TOS_AGREED", "1")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))  # < 600

# ✅ Fuerza python del container (endpoint)
PY_CONTAINER = os.environ.get("PY_CONTAINER", "/usr/local/bin/python3")

# Paths on volume
VOICES_DIR = os.path.join(RUNPOD_VOLUME_PATH, "voices")
FEMALE_REF_WAV = os.path.join(VOICES_DIR, "female_ref.wav")
MALE_REF_WAV = os.path.join(VOICES_DIR, "male_ref.wav")

# MuseTalk repo (tu ruta real según logs)
MUSE_REPO = os.environ.get("MUSE_REPO", os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"))

# En tu caso existe aquí:
MUSE_CONFIG = os.environ.get("MUSE_CONFIG", os.path.join(MUSE_REPO, "inference_config.json"))

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _now() -> float:
    return time.time()

def _tail(s: str, n: int = 1600) -> str:
    return (s or "")[-n:]

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

def _clean_env(repo_root: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["COQUI_TOS_AGREED"] = str(COQUI_TOS_AGREED)
    if repo_root:
        env["PYTHONPATH"] = repo_root + (
            ":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
        )
    return env

def _looks_like_url(s: Any) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)

def _assert_paths() -> None:
    if not os.path.isfile(PY_CONTAINER):
        raise RuntimeError(f"PY_CONTAINER not found: {PY_CONTAINER}")
    if not os.path.isdir(MUSE_REPO):
        raise RuntimeError(f"MUSE_REPO not found: {MUSE_REPO}")
    if not os.path.isfile(os.path.join(MUSE_REPO, "scripts", "inference.py")):
        raise RuntimeError(f"MuseTalk inference.py not found in: {MUSE_REPO}/scripts/inference.py")
    if not os.path.isfile(MUSE_CONFIG):
        raise RuntimeError(f"MUSE_CONFIG not found: {MUSE_CONFIG}")

# --------------------------------------------------
# Probe (para que veas SI el container ya tiene todo)
# --------------------------------------------------
def mode_scan() -> Dict[str, Any]:
    _assert_paths()
    env = _clean_env(MUSE_REPO)

    code, out = _run(
        [
            PY_CONTAINER, "-c",
            "import sys;"
            "import cv2;"
            "import mmengine;"
            "import mmcv;"
            "import mmpose;"
            "import musetalk;"
            "print('OK_ALL');"
            "print('PY', sys.executable);"
        ],
        cwd=MUSE_REPO,
        env=env,
        timeout=60,
    )
    return {
        "ok": code == 0,
        "msg": "SCAN_OK" if code == 0 else "SCAN_FAIL",
        "py": PY_CONTAINER,
        "repo_root": MUSE_REPO,
        "config": MUSE_CONFIG,
        "tail": _tail(out),
    }

def mode_echo() -> Dict[str, Any]:
    checks = {
        "py_container_exists": os.path.isfile(PY_CONTAINER),
        "muse_repo_exists": os.path.isdir(MUSE_REPO),
        "muse_infer_exists": os.path.isfile(os.path.join(MUSE_REPO, "scripts", "inference.py")),
        "muse_config_exists": os.path.isfile(MUSE_CONFIG),
        "voices_dir_exists": os.path.isdir(VOICES_DIR),
        "female_ref_exists": os.path.isfile(FEMALE_REF_WAV),
        "male_ref_exists": os.path.isfile(MALE_REF_WAV),
    }
    return {
        "ok": True,
        "msg": "ECHO_OK",
        "base": RUNPOD_VOLUME_PATH,
        "py": PY_CONTAINER,
        "repo_root": MUSE_REPO,
        "config": MUSE_CONFIG,
        "checks": checks,
    }

# --------------------------------------------------
# MuseTalk runner
# --------------------------------------------------
def _musetalk_infer(repo_root: str, config_path: str, timeout=HARD_TIMEOUT_SEC) -> Dict[str, Any]:
    _assert_paths()

    # Copia config al repo para que inference lo vea simple
    local_cfg = os.path.join(repo_root, "inference_config.json")
    try:
        shutil.copyfile(config_path, local_cfg)
    except Exception:
        pass

    env = _clean_env(repo_root)

    cmd = [
        PY_CONTAINER, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
    ]

    code, out = _run(cmd, cwd=repo_root, env=env, timeout=timeout)
    if code != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

    # busca mp4 más nuevo dentro del repo (best-effort)
    newest = None
    newest_mtime = 0.0
    for root, _, files in os.walk(repo_root):
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

def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    OJO: MuseTalk normalmente toma input/audio desde inference_config.json.
    Este worker SOLO descarga/guarda los archivos para que TU config los consuma.
    Si tu config usa rutas fijas (ej: ./data/input.mp4 y ./data/audio.wav), respétalo.

    Input:
      - video_url o video_b64
      - audio_url o audio_b64
    """
    start = _now()
    _assert_paths()

    # Workspace temporal (NO volumen)
    tmp = tempfile.mkdtemp(prefix="v2v_")
    try:
        in_mp4 = os.path.join(tmp, "input.mp4")
        wav = os.path.join(tmp, "audio.wav")

        # ---- get video ----
        if _looks_like_url(inp.get("video_url")):
            _download(inp["video_url"], in_mp4)
        elif isinstance(inp.get("video_b64"), str) and inp["video_b64"]:
            with open(in_mp4, "wb") as f:
                f.write(base64.b64decode(inp["video_b64"]))
        else:
            raise RuntimeError("Missing video_url or video_b64")

        # ---- get audio ----
        if _looks_like_url(inp.get("audio_url")):
            _download(inp["audio_url"], wav)
        elif isinstance(inp.get("audio_b64"), str) and inp["audio_b64"]:
            with open(wav, "wb") as f:
                f.write(base64.b64decode(inp["audio_b64"]))
        else:
            raise RuntimeError("Missing audio_url OR audio_b64")

        # ⚠️ Aquí NO tocamos tu inference_config.json (por tu instrucción).
        # Solo ejecutamos MuseTalk tal cual tu setup ya lo tenía funcionando.

        info = _musetalk_infer(MUSE_REPO, MUSE_CONFIG)
        elapsed = int((_now() - start) * 1000)

        return {
            "ok": True,
            "msg": "VOICE2VIDEO_OK",
            "execution_ms": elapsed,
            "repo_root": MUSE_REPO,
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
