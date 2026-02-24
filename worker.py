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

WORKER_VERSION_TAG = "v5-pydeps-mmpose-nobuildisolation-2026-02-23"

RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "30"))

# En tu endpoint real es /opt/conda/bin/python (pero usamos sys.executable por seguridad)
PY_CONTAINER = os.environ.get("PY_CONTAINER", sys.executable)

# MuseTalk repo (tu ruta real según logs)
MUSE_REPO = os.environ.get("MUSE_REPO", os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"))

# Deps persistentes en network volume
PYDEPS_DIR = os.environ.get("PYDEPS_DIR", os.path.join(RUNPOD_VOLUME_PATH, "pydeps_py310"))
PYDEPS_MARK = os.path.join(PYDEPS_DIR, ".installed_ok_mmpose")

# Puedes pinnear versión si querés:
# Ej: export MMPPOSE_SPEC="mmpose==1.3.2"
MMPPOSE_SPEC = os.environ.get("MMPPOSE_SPEC", "mmpose")

def _now() -> float:
    return time.time()

def _tail(s: str, n: int = 1800) -> str:
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

    # ✅ IMPORTANTÍSIMO:
    # Para que el subprocess (MuseTalk) vea mmpose instalado en /runpod-volume/pydeps_py310
    pypath_parts = []
    if PYDEPS_DIR:
        pypath_parts.append(PYDEPS_DIR)
    if repo_root:
        pypath_parts.append(repo_root)
    old = env.get("PYTHONPATH", "")
    if old:
        pypath_parts.append(old)

    if pypath_parts:
        env["PYTHONPATH"] = ":".join([p for p in pypath_parts if p])

    return env

def _repo_check() -> Dict[str, Any]:
    exists = os.path.isdir(MUSE_REPO)
    infer_py = os.path.join(MUSE_REPO, "scripts", "inference.py")
    return {
        "muse_repo": MUSE_REPO,
        "repo_exists": exists,
        "has_inference_py": os.path.isfile(infer_py),
        "inference_py": infer_py,
    }

def _ensure_mmpose_installed() -> Dict[str, Any]:
    """
    ✅ Instala SOLO mmpose (y deps) en PYDEPS_DIR sin tocar el entorno global.
    ✅ Usa --no-build-isolation para evitar el error:
       ModuleNotFoundError: No module named 'pip' (en build env de chumpy)
    """
    os.makedirs(PYDEPS_DIR, exist_ok=True)

    if os.path.isfile(PYDEPS_MARK):
        return {"ok": True, "already": True, "pydeps_dir": PYDEPS_DIR, "mmpose_spec": MMPPOSE_SPEC}

    env = _clean_env(None)

    # ping pip
    c0, o0 = _run([PY_CONTAINER, "-m", "pip", "--version"], env=env, timeout=SCAN_TIMEOUT_SEC)

    # ⚠️ clave: --no-build-isolation
    # (esto hace que si algo necesita setup.py, use el env donde SÍ existe pip)
    cmd = [
        PY_CONTAINER, "-m", "pip", "install",
        "--no-cache-dir",
        "--no-build-isolation",
        "--target", PYDEPS_DIR,
        MMPPOSE_SPEC
    ]
    c1, o1 = _run(cmd, env=env, timeout=HARD_TIMEOUT_SEC)

    ok = (c1 == 0)
    if ok:
        with open(PYDEPS_MARK, "w", encoding="utf-8") as f:
            f.write("ok\n")

    return {
        "ok": ok,
        "already": False,
        "pydeps_dir": PYDEPS_DIR,
        "mmpose_spec": MMPPOSE_SPEC,
        "pip_version": _tail(o0, 400),
        "install": {"code": c1, "tail": _tail(o1)},
        "cmd": " ".join(cmd),
    }

def _activate_pydeps_for_this_process() -> None:
    # Para imports dentro del worker (no solo subprocess)
    if PYDEPS_DIR and PYDEPS_DIR not in sys.path:
        sys.path.insert(0, PYDEPS_DIR)

def _import_check_in_worker() -> Dict[str, Any]:
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

def _musetalk_infer_subprocess() -> Dict[str, Any]:
    """
    Corre MuseTalk con PY_CONTAINER pero con PYTHONPATH incluyendo:
      /runpod-volume/pydeps_py310 + repo_root
    Así el subprocess sí ve mmpose.
    """
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
        "mmpose_spec": MMPPOSE_SPEC,
    }

def mode_echo() -> Dict[str, Any]:
    repo = _repo_check()

    inst = _ensure_mmpose_installed()

    _activate_pydeps_for_this_process()
    chk = _import_check_in_worker()

    return {
        "ok": True,
        "msg": "ECHO_OK",
        "worker_version": WORKER_VERSION_TAG,
        "repo": repo,
        "install": inst,
        "imports": chk,
        "py_container": PY_CONTAINER,
        "sys_executable": sys.executable,
        # para verificar que subprocess también verá PYDEPS_DIR
        "pythopath_effective_preview": _clean_env(MUSE_REPO).get("PYTHONPATH", ""),
    }

def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    start = _now()
    repo = _repo_check()
    if not repo["repo_exists"]:
        raise RuntimeError(f"MuseTalk repo not found at {MUSE_REPO}")

    inst = _ensure_mmpose_installed()
    if not inst.get("ok"):
        raise RuntimeError("Failed to install mmpose into volume pydeps.\n" + str(inst))

    _activate_pydeps_for_this_process()
    chk = _import_check_in_worker()
    if not chk.get("ok"):
        raise RuntimeError("Deps import check failed in worker.\n" + str(chk))

    # (Aquí no cambiamos tu config; MuseTalk toma rutas desde inference_config.json)
    info = _musetalk_infer_subprocess()

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
