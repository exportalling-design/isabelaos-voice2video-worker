import os
import sys
import time
import traceback
import subprocess
from typing import Any, Dict, Optional, Tuple, List

import runpod

WORKER_VERSION_TAG = "v6-skip-install-if-mmpose-ok-remove-no-use-pep517-2026-02-23"

RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "30"))

# El python real del endpoint (tu output mostró /opt/conda/bin/python)
PY_CONTAINER = os.environ.get("PY_CONTAINER", sys.executable)

# MuseTalk repo (tu ruta real)
MUSE_REPO = os.environ.get(
    "MUSE_REPO",
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk")
)

# Deps persistentes en network volume (solo si hiciera falta)
PYDEPS_DIR = os.environ.get("PYDEPS_DIR", os.path.join(RUNPOD_VOLUME_PATH, "pydeps_py310"))
PYDEPS_MARK = os.path.join(PYDEPS_DIR, ".installed_ok_mmpose")

# Intentos de versiones (override con env MMPPOSE_SPECS="mmpose==1.3.2,mmpose==1.2.0")
MMPPOSE_SPECS_ENV = os.environ.get("MMPPOSE_SPECS", "").strip()
if MMPPOSE_SPECS_ENV:
    MMPPOSE_SPECS = [s.strip() for s in MMPPOSE_SPECS_ENV.split(",") if s.strip()]
else:
    MMPPOSE_SPECS = ["mmpose==1.3.2", "mmpose==1.2.0", "mmpose==1.1.0", "mmpose"]


def _now() -> float:
    return time.time()


def _tail(s: str, n: int = 2000) -> str:
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


def _repo_check() -> Dict[str, Any]:
    infer_py = os.path.join(MUSE_REPO, "scripts", "inference.py")
    return {
        "muse_repo": MUSE_REPO,
        "repo_exists": os.path.isdir(MUSE_REPO),
        "has_inference_py": os.path.isfile(infer_py),
        "inference_py": infer_py,
    }


def _clean_env(repo_root: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"

    # Para que el subprocess vea pydeps + repo
    parts = []
    if PYDEPS_DIR:
        parts.append(PYDEPS_DIR)
    if repo_root:
        parts.append(repo_root)
    old = env.get("PYTHONPATH", "")
    if old:
        parts.append(old)
    if parts:
        env["PYTHONPATH"] = ":".join([p for p in parts if p])
    return env


def _activate_pydeps_for_this_process() -> None:
    if PYDEPS_DIR and PYDEPS_DIR not in sys.path:
        sys.path.insert(0, PYDEPS_DIR)


def _container_versions() -> Dict[str, Any]:
    env = _clean_env(None)
    code, out = _run(
        [
            PY_CONTAINER,
            "-c",
            "import sys; import mmcv, mmengine; "
            "print('PY',sys.executable); "
            "print('mmcv',mmcv.__version__); "
            "print('mmengine',mmengine.__version__)"
        ],
        env=env,
        timeout=SCAN_TIMEOUT_SEC,
    )
    return {"ok": code == 0, "code": code, "out_tail": _tail(out)}


def _quick_import_mmpose_here() -> Dict[str, Any]:
    """
    ✅ Si mmpose ya está en el container (tu caso), no se instala nada.
    """
    try:
        import mmpose  # noqa
        return {"ok": True, "msg": "OK_mmpose", "source": "container_or_sys_path"}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}


def _ensure_mmpose_available_nodeps() -> Dict[str, Any]:
    """
    ✅ NUEVO COMPORTAMIENTO:
    1) Si mmpose YA importa -> return ok=True (no toca nada)
    2) Si NO importa -> intenta instalar SOLO mmpose en PYDEPS_DIR con --no-deps
       (SIN --no-use-pep517 porque pip 26 ya no lo soporta)
    """
    # 1) si ya está, listo.
    _activate_pydeps_for_this_process()
    pre = _quick_import_mmpose_here()
    if pre.get("ok"):
        return {"ok": True, "already": True, "reason": "mmpose already importable", "precheck": pre, "pydeps_dir": PYDEPS_DIR}

    # 2) si no está, entonces sí intentamos pydeps
    os.makedirs(PYDEPS_DIR, exist_ok=True)

    # si existe mark, igual re-checkeamos (por si se quedó viejo)
    if os.path.isfile(PYDEPS_MARK):
        _activate_pydeps_for_this_process()
        post = _quick_import_mmpose_here()
        if post.get("ok"):
            return {"ok": True, "already": True, "reason": "mark present + import ok", "precheck": pre, "postcheck": post, "pydeps_dir": PYDEPS_DIR}
        # si mark existe pero sigue sin importar, seguimos a instalar

    env = _clean_env(None)

    # pip ok?
    c0, o0 = _run([PY_CONTAINER, "-m", "pip", "--version"], env=env, timeout=SCAN_TIMEOUT_SEC)
    if c0 != 0:
        return {"ok": False, "already": False, "error": "pip not available in container python", "pip_tail": _tail(o0), "pydeps_dir": PYDEPS_DIR}

    tried = []
    for spec in MMPPOSE_SPECS:
        cmd = [
            PY_CONTAINER, "-m", "pip", "install",
            "--no-cache-dir",
            "--no-deps",
            "--target", PYDEPS_DIR,
            "--no-build-isolation",
            spec
        ]
        c1, o1 = _run(cmd, env=env, timeout=HARD_TIMEOUT_SEC)
        tried.append({"spec": spec, "code": c1, "tail": _tail(o1)})

        if c1 == 0:
            # activar y probar import
            _activate_pydeps_for_this_process()
            post = _quick_import_mmpose_here()
            if post.get("ok"):
                with open(PYDEPS_MARK, "w", encoding="utf-8") as f:
                    f.write(spec + "\n")
                return {
                    "ok": True,
                    "already": False,
                    "pydeps_dir": PYDEPS_DIR,
                    "picked": spec,
                    "tried": tried,
                    "precheck": pre,
                    "postcheck": post,
                }

    return {
        "ok": False,
        "already": False,
        "pydeps_dir": PYDEPS_DIR,
        "picked": None,
        "tried": tried,
        "precheck": pre,
    }


def _import_check_in_worker() -> Dict[str, Any]:
    info = {
        "py_container": PY_CONTAINER,
        "sys_executable": sys.executable,
        "pydeps_dir": PYDEPS_DIR,
        "pydeps_active": (sys.path[0] == PYDEPS_DIR if sys.path else False),
    }

    try:
        import cv2  # noqa
        import mmcv  # noqa
        import mmengine  # noqa
        info["core"] = {"ok": True, "msg": "OK_cv2_mmcv_mmengine"}
    except Exception as e:
        info["core"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    try:
        import mmpose  # noqa
        info["mmpose"] = {"ok": True, "msg": "OK_mmpose"}
    except Exception as e:
        info["mmpose"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    info["ok"] = bool(info.get("core", {}).get("ok") and info.get("mmpose", {}).get("ok"))
    return info


def _musetalk_infer_subprocess() -> Dict[str, Any]:
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
        "mmpose_specs": MMPPOSE_SPECS,
        "versions": _container_versions(),
    }


def mode_echo() -> Dict[str, Any]:
    repo = _repo_check()
    versions = _container_versions()

    # ✅ Solo “ensure” si hace falta (si ya importa, no instala nada)
    inst = _ensure_mmpose_available_nodeps()

    _activate_pydeps_for_this_process()
    chk = _import_check_in_worker()

    return {
        "ok": True,
        "msg": "ECHO_OK",
        "worker_version": WORKER_VERSION_TAG,
        "repo": repo,
        "versions": versions,
        "ensure_mmpose": inst,
        "imports": chk,
        "py_container": PY_CONTAINER,
        "sys_executable": sys.executable,
        "pythopath_effective_preview": _clean_env(MUSE_REPO).get("PYTHONPATH", ""),
    }


def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    start = _now()
    repo = _repo_check()
    if not repo["repo_exists"]:
        raise RuntimeError(f"MuseTalk repo not found at {MUSE_REPO}")

    versions = _container_versions()

    # ✅ Si mmpose ya está, no toca nada; si no, intenta pydeps
    ensure = _ensure_mmpose_available_nodeps()
    if not ensure.get("ok"):
        raise RuntimeError("mmpose not available (and install to pydeps failed).\n" + str(ensure))

    _activate_pydeps_for_this_process()
    chk = _import_check_in_worker()
    if not chk.get("ok"):
        raise RuntimeError("Deps import check failed in worker.\n" + str(chk))

    # Aquí ya corre MuseTalk con PYTHONPATH incluyendo pydeps + repo
    info = _musetalk_infer_subprocess()

    elapsed = int((_now() - start) * 1000)
    return {
        "ok": True,
        "msg": "VOICE2VIDEO_OK",
        "worker_version": WORKER_VERSION_TAG,
        "execution_ms": elapsed,
        "repo": repo,
        "versions": versions,
        "ensure_mmpose": ensure,
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
