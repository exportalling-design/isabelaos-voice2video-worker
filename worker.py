import os
import sys
import time
import traceback
import subprocess
from typing import Any, Dict, Optional, Tuple, List

import runpod

# ✅ bump version
WORKER_VERSION_TAG = "v10-autoinstall-librosa-to-pydeps-2026-02-23"

RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "30"))

# Python real del endpoint (tu output mostró /opt/conda/bin/python)
PY_CONTAINER = os.environ.get("PY_CONTAINER", sys.executable)

# MuseTalk repo (tu ruta real)
MUSE_REPO = os.environ.get(
    "MUSE_REPO",
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk")
)

# Deps persistentes en network volume
PYDEPS_DIR = os.environ.get("PYDEPS_DIR", os.path.join(RUNPOD_VOLUME_PATH, "pydeps_py310"))

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

def _pip_ok() -> Dict[str, Any]:
    env = _clean_env(None)
    c0, o0 = _run([PY_CONTAINER, "-m", "pip", "--version"], env=env, timeout=SCAN_TIMEOUT_SEC)
    return {"ok": c0 == 0, "code": c0, "tail": _tail(o0)}

def _pip_install_target(spec: str, with_deps: bool) -> Dict[str, Any]:
    """
    Instala en PYDEPS_DIR.
    - with_deps=True para paquetes con dependencias (transformers/librosa).
    - ✅ NO usa --no-use-pep517 (tu pip 26 no lo acepta)
    """
    os.makedirs(PYDEPS_DIR, exist_ok=True)

    env = _clean_env(None)
    cmd = [
        PY_CONTAINER, "-m", "pip", "install",
        "--no-cache-dir",
        "--upgrade",
        "--target", PYDEPS_DIR,
        "--no-build-isolation",
    ]
    if not with_deps:
        cmd.append("--no-deps")
    cmd.append(spec)

    code, out = _run(cmd, env=env, timeout=HARD_TIMEOUT_SEC)
    return {"spec": spec, "with_deps": with_deps, "code": code, "tail": _tail(out)}

def _import_check(module: str) -> Tuple[bool, str]:
    try:
        __import__(module)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def _ensure_modules() -> Dict[str, Any]:
    """
    Asegura módulos mínimos que MuseTalk está pidiendo desde scripts/inference.py
    en tu runtime actual.
    """
    MODULE_SPECS = [
        {"name": "mmpose", "spec": os.environ.get("SPEC_MMPOSE", "mmpose"), "with_deps": False},
        {"name": "omegaconf", "spec": os.environ.get("SPEC_OMEGACONF", "omegaconf==2.3.0"), "with_deps": False},
        {"name": "hydra", "spec": os.environ.get("SPEC_HYDRA", "hydra-core==1.3.2"), "with_deps": False},
        {"name": "transformers", "spec": os.environ.get("SPEC_TRANSFORMERS", "transformers==4.38.2"), "with_deps": True},

        # ✅ NUEVO: requerido por musetalk/utils/audio_processor.py
        # (instalar con deps)
        {"name": "librosa", "spec": os.environ.get("SPEC_LIBROSA", "librosa==0.10.2.post1"), "with_deps": True},
    ]

    # pip check
    pipv = _pip_ok()
    if not pipv["ok"]:
        return {"ok": False, "error": "pip not available", "pip": pipv}

    ensured = {}
    installs = []

    _activate_pydeps_for_this_process()

    for m in MODULE_SPECS:
        name = m["name"]
        spec = m["spec"]
        with_deps = bool(m["with_deps"])

        # hydra se importa como "hydra"
        import_name = "hydra" if name == "hydra" else name

        ok, _err = _import_check(import_name)
        if ok:
            ensured[name] = {"ok": True, "already": True, "pip_spec": spec, "reason": f"{name} import ok"}
            continue

        res = _pip_install_target(spec, with_deps=with_deps)
        installs.append(res)

        # re-check import
        _activate_pydeps_for_this_process()
        ok2, err2 = _import_check(import_name)
        if ok2:
            ensured[name] = {"ok": True, "already": False, "pip_spec": spec, "reason": "installed -> import ok"}
        else:
            ensured[name] = {"ok": False, "already": False, "pip_spec": spec, "error": err2}

    all_ok = all(v.get("ok") for v in ensured.values())
    return {"ok": all_ok, "pip": pipv, "ensure": ensured, "installs": installs, "pydeps_dir": PYDEPS_DIR}

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

    # omegaconf
    try:
        from omegaconf import OmegaConf  # noqa
        info["omegaconf"] = {"ok": True, "msg": "OK_omegaconf"}
    except Exception as e:
        info["omegaconf"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    # transformers
    try:
        import transformers  # noqa
        info["transformers"] = {"ok": True, "msg": "OK_transformers"}
    except Exception as e:
        info["transformers"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    # ✅ librosa
    try:
        import librosa  # noqa
        info["librosa"] = {"ok": True, "msg": "OK_librosa"}
    except Exception as e:
        info["librosa"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    info["ok"] = bool(
        info.get("core", {}).get("ok")
        and info.get("mmpose", {}).get("ok")
        and info.get("omegaconf", {}).get("ok")
        and info.get("transformers", {}).get("ok")
        and info.get("librosa", {}).get("ok")
    )
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

def mode_echo() -> Dict[str, Any]:
    repo = _repo_check()

    ensured = _ensure_modules()

    _activate_pydeps_for_this_process()
    chk = _import_check_in_worker()

    return {
        "ok": True,
        "msg": "ECHO_OK",
        "worker_version": WORKER_VERSION_TAG,
        "repo": repo,
        "ensure": ensured,
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

    ensured = _ensure_modules()
    if not ensured.get("ok"):
        raise RuntimeError("Missing deps after ensure().\n" + str(ensured))

    _activate_pydeps_for_this_process()
    chk = _import_check_in_worker()
    if not chk.get("ok"):
        raise RuntimeError("Deps import check failed in worker.\n" + str(chk))

    info = _musetalk_infer_subprocess()
    elapsed = int((_now() - start) * 1000)

    return {
        "ok": True,
        "msg": "VOICE2VIDEO_OK",
        "worker_version": WORKER_VERSION_TAG,
        "execution_ms": elapsed,
        "repo": repo,
        "ensure": ensured,
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

        mode = str(inp.get("mode", "echo")).strip().lower()

        if mode in ("echo",):
            return mode_echo()
        if mode in ("voice2video", "v2v"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
