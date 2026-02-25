import os
import sys
import time
import traceback
import subprocess
import shutil
from typing import Any, Dict, Optional, Tuple

import runpod

WORKER_VERSION_TAG = "v12-add-diffusers-pin-numpy-constraints-2026-02-24"

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

# ---- Pins / Specs ----
SPEC_NUMPY = os.environ.get("SPEC_NUMPY", "numpy==1.26.4")
SPEC_MMPOSE = os.environ.get("SPEC_MMPOSE", "mmpose")
SPEC_OMEGACONF = os.environ.get("SPEC_OMEGACONF", "omegaconf==2.3.0")
SPEC_HYDRA = os.environ.get("SPEC_HYDRA", "hydra-core==1.3.2")
SPEC_TRANSFORMERS = os.environ.get("SPEC_TRANSFORMERS", "transformers==4.38.2")
SPEC_LIBROSA = os.environ.get("SPEC_LIBROSA", "librosa==0.10.2.post1")
SPEC_EINOPS = os.environ.get("SPEC_EINOPS", "einops==0.7.0")
# ✅ NUEVO: diffusers (lo que te falló ahora)
SPEC_DIFFUSERS = os.environ.get("SPEC_DIFFUSERS", "diffusers==0.27.2")

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

def _import_check(module: str) -> Tuple[bool, str]:
    try:
        __import__(module)
        return True, "OK"
    except Exception as e:
        return False, str(e)

def _write_constraints() -> str:
    """
    Constraints para evitar que pip suba numpy a 2.x dentro de PYDEPS_DIR.
    """
    os.makedirs(PYDEPS_DIR, exist_ok=True)
    path = os.path.join(PYDEPS_DIR, "_constraints.txt")
    content = "\n".join([
        "numpy<2",
        "numpy==1.26.4",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def _purge_numpy_from_pydeps() -> Dict[str, Any]:
    """
    Borra cualquier numpy instalado en PYDEPS_DIR (1.x o 2.x) para que el reinstall sea limpio.
    """
    os.makedirs(PYDEPS_DIR, exist_ok=True)
    removed = []
    for name in os.listdir(PYDEPS_DIR):
        low = name.lower()
        if low == "numpy" or low.startswith("numpy-") or low.startswith("numpy.") or low.startswith("numpy_"):
            p = os.path.join(PYDEPS_DIR, name)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
                removed.append(name)
            except Exception:
                pass
        if low.startswith("numpy-") and low.endswith(".dist-info"):
            p = os.path.join(PYDEPS_DIR, name)
            try:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(name)
            except Exception:
                pass
        if low == "numpy.libs":
            p = os.path.join(PYDEPS_DIR, name)
            try:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(name)
            except Exception:
                pass
    return {"ok": True, "removed": removed}

def _pip_install_target(spec: str, with_deps: bool, constraints_path: Optional[str] = None) -> Dict[str, Any]:
    os.makedirs(PYDEPS_DIR, exist_ok=True)
    env = _clean_env(None)

    cmd = [
        PY_CONTAINER, "-m", "pip", "install",
        "--no-cache-dir",
        "--target", PYDEPS_DIR,
        "--upgrade",
        "--no-build-isolation",
    ]
    if constraints_path:
        cmd += ["-c", constraints_path]
    if not with_deps:
        cmd.append("--no-deps")
    cmd.append(spec)

    code, out = _run(cmd, env=env, timeout=HARD_TIMEOUT_SEC)
    return {"spec": spec, "with_deps": with_deps, "code": code, "tail": _tail(out)}

def _ensure_numpy_pinned() -> Dict[str, Any]:
    """
    Asegura numpy==1.26.4 en PYDEPS_DIR, purgando cualquier numpy previo (incluye 2.x).
    """
    pipv = _pip_ok()
    if not pipv["ok"]:
        return {"ok": False, "error": "pip not available", "pip": pipv}

    constraints = _write_constraints()
    pur = _purge_numpy_from_pydeps()

    res = _pip_install_target(SPEC_NUMPY, with_deps=False, constraints_path=constraints)

    _activate_pydeps_for_this_process()
    ok, err = _import_check("numpy")
    ver = None
    if ok:
        try:
            import numpy as np  # noqa
            ver = getattr(np, "__version__", None)
        except Exception:
            pass

    return {
        "ok": ok and (ver is not None) and ver.startswith("1."),
        "pip": pipv,
        "constraints": constraints,
        "purge": pur,
        "install": res,
        "numpy_version": ver,
        "numpy_import_error": None if ok else err
    }

def _ensure_modules() -> Dict[str, Any]:
    """
    Módulos que MuseTalk ya pidió en tus logs.
    Importante:
    - Primero fijamos numpy==1.26.4 (evita crash/torch warning con numpy2)
    - Paquetes con deps (transformers/librosa/diffusers) se instalan con constraints numpy<2
    - mmpose sin deps
    """
    pipv = _pip_ok()
    if not pipv["ok"]:
        return {"ok": False, "error": "pip not available", "pip": pipv}

    _activate_pydeps_for_this_process()

    numpy_fix = _ensure_numpy_pinned()
    if not numpy_fix.get("ok"):
        return {"ok": False, "error": "failed to pin numpy<2", "numpy_fix": numpy_fix, "pip": pipv}

    constraints = numpy_fix.get("constraints") or _write_constraints()

    module_plan = [
        # sin deps
        {"name": "omegaconf", "import": "omegaconf", "spec": SPEC_OMEGACONF, "with_deps": False},
        {"name": "hydra", "import": "hydra", "spec": SPEC_HYDRA, "with_deps": False},
        {"name": "einops", "import": "einops", "spec": SPEC_EINOPS, "with_deps": False},
        {"name": "mmpose", "import": "mmpose", "spec": SPEC_MMPOSE, "with_deps": False},

        # con deps (pero con constraints numpy<2)
        {"name": "transformers", "import": "transformers", "spec": SPEC_TRANSFORMERS, "with_deps": True},
        {"name": "librosa", "import": "librosa", "spec": SPEC_LIBROSA, "with_deps": True},
        # ✅ NUEVO: diffusers (te estaba faltando)
        {"name": "diffusers", "import": "diffusers", "spec": SPEC_DIFFUSERS, "with_deps": True},
    ]

    ensured = {}
    installs = []

    for m in module_plan:
        name = m["name"]
        mod_import = m["import"]
        spec = m["spec"]
        with_deps = bool(m["with_deps"])

        ok, err = _import_check(mod_import)
        if ok:
            ensured[name] = {"ok": True, "already": True, "pip_spec": spec, "reason": f"{name} import ok"}
            continue

        res = _pip_install_target(
            spec,
            with_deps=with_deps,
            constraints_path=constraints if with_deps else None
        )
        installs.append(res)

        _activate_pydeps_for_this_process()
        ok2, err2 = _import_check(mod_import)
        if ok2:
            ensured[name] = {"ok": True, "already": False, "pip_spec": spec, "reason": "installed -> import ok"}
        else:
            ensured[name] = {"ok": False, "already": False, "pip_spec": spec, "error": err2, "pip_tail": res.get("tail")}

    all_ok = all(v.get("ok") for v in ensured.values())
    return {
        "ok": all_ok,
        "pip": pipv,
        "numpy_fix": numpy_fix,
        "constraints": constraints,
        "ensure": ensured,
        "installs": installs,
        "pydeps_dir": PYDEPS_DIR
    }

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

    # numpy (para confirmar que quedó 1.x)
    try:
        import numpy as np  # noqa
        info["numpy"] = {"ok": True, "msg": "OK_numpy", "version": getattr(np, "__version__", None)}
    except Exception as e:
        info["numpy"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

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

    # librosa
    try:
        import librosa  # noqa
        info["librosa"] = {"ok": True, "msg": "OK_librosa"}
    except Exception as e:
        info["librosa"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    # einops
    try:
        import einops  # noqa
        info["einops"] = {"ok": True, "msg": "OK_einops"}
    except Exception as e:
        info["einops"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    # ✅ NUEVO: diffusers
    try:
        import diffusers  # noqa
        info["diffusers"] = {"ok": True, "msg": "OK_diffusers"}
    except Exception as e:
        info["diffusers"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    info["ok"] = bool(
        info.get("core", {}).get("ok")
        and info.get("numpy", {}).get("ok")
        and str(info.get("numpy", {}).get("version", "")).startswith("1.")
        and info.get("mmpose", {}).get("ok")
        and info.get("omegaconf", {}).get("ok")
        and info.get("transformers", {}).get("ok")
        and info.get("librosa", {}).get("ok")
        and info.get("einops", {}).get("ok")
        and info.get("diffusers", {}).get("ok")
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

    # ensure deps
    ensured = _ensure_modules()
    if not ensured.get("ok"):
        raise RuntimeError("Missing deps after ensure().\n" + str(ensured))

    _activate_pydeps_for_this_process()
    chk = _import_check_in_worker()
    if not chk.get("ok"):
        raise RuntimeError("Deps import check failed in worker.\n" + str(chk))

    # run musetalk
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

        if mode == "echo":
            return mode_echo()
        if mode in ("voice2video", "v2v"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
