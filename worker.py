# /app/worker.py
# RunPod Serverless Worker — MuseTalk runner (NO instala nada)
# ✅ Hardcode: usa el python que ya te funciona en el POD:
#    /workspace/musetalk_ok/bin/python
# ✅ Repo: /runpod-volume/volume_old/MuseTalk (ajustable)
# Modes: echo, run

import os
import time
import base64
import shutil
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Tuple, Optional

import runpod

HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))

# 🔥 FORZADO: python bueno (el que vos probaste "OK todo")
FORCED_PY = os.environ.get("MUSE_PY", "/workspace/musetalk_ok/bin/python")

# Repo MuseTalk (en tu output sale aquí)
MUSE_REPO = os.environ.get("MUSE_REPO", "/runpod-volume/volume_old/MuseTalk")

def _now() -> float:
    return time.time()

def _tail(s: str, n: int = 2000) -> str:
    return (s or "")[-n:]

def _clean_env(repo_root: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if repo_root:
        env["PYTHONPATH"] = repo_root + (
            ":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
        )
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
    with urllib.request.urlopen(req, timeout=60) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)

def mode_echo() -> Dict[str, Any]:
    # prueba imports con el python forzado
    env = _clean_env(MUSE_REPO)
    c1, o1 = _run([FORCED_PY, "-c", "import cv2; import mmcv; import mmengine; import mmpose; print('OK_ALL')"], env=env, timeout=40)
    return {
        "ok": True,
        "mode": "echo",
        "forced_python": FORCED_PY,
        "muse_repo": MUSE_REPO,
        "import_ok": (c1 == 0),
        "import_tail": _tail(o1),
        "exists": {
            "python": os.path.isfile(FORCED_PY),
            "repo": os.path.isdir(MUSE_REPO),
            "inference": os.path.isfile(os.path.join(MUSE_REPO, "scripts", "inference.py")),
            "cfg": os.path.isfile(os.path.join(MUSE_REPO, "inference_config.json")),
        }
    }

def mode_run(inp: Dict[str, Any]) -> Dict[str, Any]:
    if not os.path.isdir(MUSE_REPO):
        raise RuntimeError(f"MuseTalk repo not found: {MUSE_REPO}")
    if not os.path.isfile(os.path.join(MUSE_REPO, "scripts", "inference.py")):
        raise RuntimeError("MuseTalk scripts/inference.py not found")
    if not os.path.isfile(os.path.join(MUSE_REPO, "inference_config.json")):
        raise RuntimeError("MuseTalk inference_config.json not found in repo (required)")

    tmp = tempfile.mkdtemp(prefix="musetalk_")
    start = _now()
    try:
        in_mp4 = os.path.join(tmp, "input.mp4")
        in_wav = os.path.join(tmp, "audio.wav")

        # video
        if _looks_like_url(inp.get("video_url")):
            _download(inp["video_url"], in_mp4)
        elif isinstance(inp.get("video_b64"), str) and inp["video_b64"]:
            with open(in_mp4, "wb") as f:
                f.write(base64.b64decode(inp["video_b64"]))
        else:
            raise RuntimeError("Missing video_url or video_b64")

        # audio
        if _looks_like_url(inp.get("audio_url")):
            _download(inp["audio_url"], in_wav)
        elif isinstance(inp.get("audio_b64"), str) and inp["audio_b64"]:
            with open(in_wav, "wb") as f:
                f.write(base64.b64decode(inp["audio_b64"]))
        else:
            raise RuntimeError("Missing audio_url or audio_b64")

        # MuseTalk normalmente lee input/audio desde inference_config.json
        # (No tocamos tu config aquí para no romper tu setup)

        env = _clean_env(MUSE_REPO)

        cmd = [
            FORCED_PY, "-u", "scripts/inference.py",
            "--inference_config", "inference_config.json",
        ]

        code, out = _run(cmd, cwd=MUSE_REPO, env=env, timeout=HARD_TIMEOUT_SEC)
        if code != 0:
            raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

        # output guess
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
            "mode": "run",
            "forced_python": FORCED_PY,
            "muse_repo": MUSE_REPO,
            "execution_ms": int((_now() - start) * 1000),
            "output_mp4_guess": newest,
            "log_tail": _tail(out),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = event.get("input") if isinstance(event, dict) else None
        if not isinstance(inp, dict):
            return {"ok": False, "error": "Missing or invalid input"}

        mode = str(inp.get("mode", "echo")).strip().lower()
        if mode == "echo":
            return mode_echo()
        if mode in ("run", "voice2video", "v2v"):
            return mode_run(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
