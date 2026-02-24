import os
import sys
import time
import base64
import shutil
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, Tuple

import runpod

# ✅ Si ves esto en la salida, es ESTE worker (sin dudas)
WORKER_VERSION_TAG = "v4-pydeps-volume-numpy<2-mmpose-2026-02-23"

RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "30"))

# Container python real (en tu endpoint es /opt/conda/bin/python)
PY_CONTAINER = os.environ.get("PY_CONTAINER", sys.executable)

# MuseTalk repo en tu volumen (según tus logs)
MUSE_REPO = os.environ.get("MUSE_REPO", os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"))

# ✅ Carpeta persistente de deps en network volume
PYDEPS_DIR = os.environ.get("PYDEPS_DIR", os.path.join(RUNPOD_VOLUME_PATH, "pydeps_py310"))
PYDEPS_MARK = os.path.join(PYDEPS_DIR, ".installed_ok")

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

def _repo_check() -> Dict[str, Any]:
    exists = os.path.isdir(MUSE_REPO)
    infer_py = os.path.join(MUSE_REPO, "scripts", "inference.py")
    return {
        "muse_repo": MUSE_REPO,
        "repo_exists": exists,
        "has_inference_py": os.path.isfile(infer_py),
        "inference_py": infer_py,
    }

def _ensure_pydeps_installed() -> Dict[str, Any]:
    """
    ✅ Instala en /runpod-volume/pydeps_py310 (persistente)
    - numpy<2 (para arreglar warning/errores de numpy2 con torch/mmcv)
    - mmpose (lo que te falta)
    Solo se ejecuta 1 vez por volumen (marca .installed_ok)
    """
    os.makedirs(PYDEPS_DIR, exist_ok=True)

    if os.path.isfile(PYDEPS_MARK):
        return {"ok": True, "already": True, "pydeps_dir": PYDEPS_DIR}

    env = _clean_env(None)

    # 1) pip básico
    c0, o0 = _run([PY_CONTAINER, "-m", "pip", "--version"], env=env, timeout=SCAN_TIMEOUT_SEC)

    # 2) numpy<2 a target
    c1, o1 = _run(
        [PY_CONTAINER, "-m", "pip", "install", "--no-cache-dir", "--target", PYDEPS_DIR, "numpy<2"],
        env=env,
        timeout=HARD_TIMEOUT_SEC,
    )

    # 3) mmpose a target (sin tocar tu entorno global del container)
    #    - Esto puede traer deps python puros. Si alguna dep binaria falla, lo veremos en el log.
    c2, o2 = _run(
        [PY_CONTAINER, "-m", "pip", "install", "--no-cache-dir", "--target", PYDEPS_DIR, "mmpose"],
        env=env,
        timeout=HARD_TIMEOUT_SEC,
    )

    ok = (c1 == 0 and c2 == 0)
    if ok:
        with open(PYDEPS_MARK, "w", encoding="utf-8") as f:
            f.write("ok\n")

    return {
        "ok": ok,
        "already": False,
        "pydeps_dir": PYDEPS_DIR,
        "pip_version": _tail(o0, 400),
        "numpy_install": {"code": c1, "tail": _tail(o1)},
        "mmpose_install": {"code": c2, "tail": _tail(o2)},
    }

def _activate_pydeps() -> None:
    """
    ✅ Mete PYDEPS_DIR al inicio del sys.path para que:
    - use numpy<2 del volumen
    - encuentre mmpose del volumen
    """
    if PYDEPS_DIR not in sys.path:
        sys.path.insert(0, PYDEPS_DIR)

def _container_import_check() -> Dict[str, Any]:
    """
    - Corre import check desde el MISMO proceso (no con -c) ya con sys.path modificado.
    """
    info = {
        "py_container": PY_CONTAINER,
        "sys_executable": sys.executable,
        "pydeps_dir": PYDEPS_DIR,
        "pydeps_active": (sys.path[0] == PYDEPS_DIR if sys.path else False),
    }

    # core
    try:
        import cv2  # noqa
        import mmcv  # noqa
        import mmengine  # noqa
        info["core"] = {"ok": True, "msg": "OK_cv2_mmcv_mmengine"}
    except Exception as e:
        info["core"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    # mmpose
    try:
        import mmpose  # noqa
        info["mmpose"] = {"ok": True, "msg": "OK_mmpose"}
    except Exception as e:
        info["mmpose"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    info["ok"] = bool(info.get("core", {}).get("ok") and info.get("mmpose", {}).get("ok"))
    return info

def _musetalk_infer() -> Dict[str, Any]:
    if not os.path.isdir(MUSE_REPO):
        raise RuntimeError(f"MuseTalk repo not found: {MUSE_REPO}")
    if not os.path.isfile(os.path.join(MUSE_REPO, "scripts", "inference.py")):
        raise RuntimeError("MuseTalk scripts/inference.py not found in repo")

    env = _clean_env(MUSE_REPO)

    cmd = [PY_CONTAINER, "-u", "scripts/inference.py", "--inference_config", "inference_config.json"]
    code, out = _run(cmd, cwd=MUSE_REPO, env=env, timeout=HARD_TIMEOUT_SEC)
    if code != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

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

    return {"ok": True, "python_used": PY_CONTAINER, "output_mp4_guess": newest, "log_tail": _tail(out)}

def mode_scan() -> Dict[str, Any]:
    return {
        "ok": True,
        "msg": "SCAN_OK",
        "worker_version": WORKER_VERSION_TAG,
        "repo": _repo_check(),
        "py_container": PY_CONTAINER,
        "sys_executable": sys.executable,
        "pydeps_dir": PYDEPS_DIR,
    }

def mode_echo() -> Dict[str, Any]:
    repo = _repo_check()

    # 1) instala deps persistentes (1 vez)
    inst = _ensure_pydeps_installed()

    # 2) activa sys.path
    _activate_pydeps()

    # 3) importa desde este proceso (ya con pydeps)
    chk = _container_import_check()

    return {
        "ok": True,
        "msg": "ECHO_OK",
        "worker_version": WORKER_VERSION_TAG,
        "repo": repo,
        "install": inst,
        "imports": chk,
        "py_container": PY_CONTAINER,
        "sys_executable": sys.executable,
    }

def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    start = _now()
    repo = _repo_check()
    if not repo["repo_exists"]:
        raise RuntimeError(f"MuseTalk repo not found at {MUSE_REPO}")

    inst = _ensure_pydeps_installed()
    if not inst.get("ok"):
        raise RuntimeError("Failed to install required deps into volume pydeps.\n" + str(inst))

    _activate_pydeps()
    chk = _container_import_check()
    if not chk.get("ok"):
        raise RuntimeError("Deps import check failed.\n" + str(chk))

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

        info = _musetalk_infer()
        elapsed = int((_now() - start) * 1000)

        return {
            "ok": True,
            "msg": "VOICE2VIDEO_OK",
            "worker_version": WORKER_VERSION_TAG,
            "execution_ms": elapsed,
            "repo": repo,
            "install": inst,
            "imports": chk,
            "python_used": info.get("python_used"),
            "output_mp4_guess": info.get("output_mp4_guess"),
            "log_tail": info.get("log_tail"),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

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
