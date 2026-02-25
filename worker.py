import os
import sys
import time
import traceback
import subprocess
import shutil
import importlib
from typing import Any, Dict, Optional, Tuple

import runpod

WORKER_VERSION_TAG = "v22-mmengine-pkg_resources-location-fix-2026-02-25"

RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "30"))

PY_CONTAINER = os.environ.get("PY_CONTAINER", sys.executable)

MUSE_REPO = os.environ.get(
    "MUSE_REPO",
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk")
)

PYDEPS_DIR = os.environ.get("PYDEPS_DIR", os.path.join(RUNPOD_VOLUME_PATH, "pydeps_py310"))

# ---- Pins / Specs ----
SPEC_NUMPY = os.environ.get("SPEC_NUMPY", "numpy==1.26.4")
SPEC_MMPOSE = os.environ.get("SPEC_MMPOSE", "mmpose")

SPEC_OMEGACONF = os.environ.get("SPEC_OMEGACONF", "omegaconf==2.3.0")
SPEC_HYDRA = os.environ.get("SPEC_HYDRA", "hydra-core==1.3.2")
SPEC_TRANSFORMERS = os.environ.get("SPEC_TRANSFORMERS", "transformers==4.38.2")
SPEC_LIBROSA = os.environ.get("SPEC_LIBROSA", "librosa==0.10.2.post1")
SPEC_EINOPS = os.environ.get("SPEC_EINOPS", "einops==0.7.0")

SPEC_DIFFUSERS = os.environ.get("SPEC_DIFFUSERS", "diffusers==0.27.2")
SPEC_HF_HUB = os.environ.get("SPEC_HF_HUB", "huggingface_hub==0.20.3")

SPEC_XTCOCO = os.environ.get("SPEC_XTCOCO", "xtcocotools==1.13.0")
SPEC_MUNKRES = os.environ.get("SPEC_MUNKRES", "munkres==1.1.4")
SPEC_MMDET = os.environ.get("SPEC_MMDET", "mmdet==3.3.0")
SPEC_SETUPTOOLS = os.environ.get("SPEC_SETUPTOOLS", "setuptools==82.0.0")


def _now() -> float:
    return time.time()


def _tail(s: str, n: int = 2400) -> str:
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
    os.makedirs(PYDEPS_DIR, exist_ok=True)
    path = os.path.join(PYDEPS_DIR, "_constraints.txt")
    content = "\n".join([
        "numpy<2",
        "numpy==1.26.4",
        "huggingface_hub==0.20.3",
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def _purge_prefix(prefix: str) -> Dict[str, Any]:
    os.makedirs(PYDEPS_DIR, exist_ok=True)
    removed = []
    for name in os.listdir(PYDEPS_DIR):
        low = name.lower()
        if low == prefix.lower() or low.startswith(prefix.lower() + "-") or low.startswith(prefix.lower() + "."):
            p = os.path.join(PYDEPS_DIR, name)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
                removed.append(name)
            except Exception:
                pass
    return {"ok": True, "removed": removed}


def _purge_numpy_from_pydeps() -> Dict[str, Any]:
    os.makedirs(PYDEPS_DIR, exist_ok=True)
    removed = []
    for name in os.listdir(PYDEPS_DIR):
        low = name.lower()
        if low == "numpy" or low.startswith("numpy-") or low == "numpy.libs":
            p = os.path.join(PYDEPS_DIR, name)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
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
    pipv = _pip_ok()
    if not pipv["ok"]:
        return {"ok": False, "error": "pip not available", "pip": pipv}

    os.makedirs(PYDEPS_DIR, exist_ok=True)
    constraints = _write_constraints()
    pur = _purge_numpy_from_pydeps()

    res = _pip_install_target(SPEC_NUMPY, with_deps=False, constraints_path=constraints)

    _activate_pydeps_for_this_process()
    importlib.invalidate_caches()

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
        "constraints": constraints,
        "purge": pur,
        "install": res,
        "numpy_version": ver,
        "numpy_import_error": None if ok else err
    }


def _ensure_hf_hub_compat(constraints: str) -> Dict[str, Any]:
    _activate_pydeps_for_this_process()
    importlib.invalidate_caches()

    try:
        import huggingface_hub as hfh  # noqa
        has_cd = hasattr(hfh, "cached_download")
        ver = getattr(hfh, "__version__", None)
        if has_cd and ver == "0.20.3":
            return {"ok": True, "has_cached_download": True, "hub_version": ver, "reason": "already compatible"}
    except Exception:
        pass

    pur = _purge_prefix("huggingface_hub")
    res = _pip_install_target(SPEC_HF_HUB, with_deps=True, constraints_path=constraints)

    _activate_pydeps_for_this_process()
    importlib.invalidate_caches()

    try:
        import huggingface_hub as hfh2  # noqa
        has_cd2 = hasattr(hfh2, "cached_download")
        ver2 = getattr(hfh2, "__version__", None)
        return {
            "ok": bool(has_cd2 and ver2 == "0.20.3"),
            "purge": pur,
            "install": res,
            "has_cached_download": has_cd2,
            "hub_version": ver2,
            "hub_import_error": None
        }
    except Exception as e:
        return {
            "ok": False,
            "purge": pur,
            "install": res,
            "has_cached_download": False,
            "hub_version": None,
            "hub_import_error": str(e)
}

def _write_pkg_resources_shim(pydeps_dir: str) -> Dict[str, Any]:
    """
    Shim mínimo de pkg_resources compatible con mmengine:
    - get_distribution(name) -> objeto con .location
    - DistributionNotFound
    """
    os.makedirs(os.path.join(pydeps_dir, "pkg_resources"), exist_ok=True)
    path = os.path.join(pydeps_dir, "pkg_resources", "__init__.py")

    shim = r'''
# --- pkg_resources shim for mmengine/mmpose ---
from __future__ import annotations

import importlib.metadata as _im
from dataclasses import dataclass

class DistributionNotFound(Exception):
    pass

@dataclass
class _DistWrap:
    _dist: object
    project_name: str
    version: str
    location: str

def _dist_location(d):
    # importlib.metadata.PathDistribution -> no .location
    # mmengine expects .location like pkg_resources Distribution
    try:
        return str(d.locate_file(""))
    except Exception:
        return str(getattr(d, "_path", ""))  # fallback

def get_distribution(name: str):
    try:
        d = _im.distribution(name)
    except Exception as e:
        raise DistributionNotFound(str(e))
    pname = ""
    try:
        pname = d.metadata.get("Name") or d.metadata.get("name") or name
    except Exception:
        pname = name
    return _DistWrap(
        _dist=d,
        project_name=pname,
        version=getattr(d, "version", "0"),
        location=_dist_location(d),
    )
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(shim)

    return {"ok": True, "msg": "pkg_resources shim (mmengine-compatible) written", "path": path}


def _ensure_pkg_resources(constraints: str) -> Dict[str, Any]:
    """
    mmengine -> get_installed_path usa pkg_resources.get_distribution().location
    En algunos containers no existe pkg_resources (setuptools no viene).
    1) instalar setuptools en PYDEPS_DIR
    2) si sigue sin existir pkg_resources, escribir shim compatible con mmengine
    """
    os.makedirs(PYDEPS_DIR, exist_ok=True)
    _activate_pydeps_for_this_process()
    importlib.invalidate_caches()

    # (A) intento directo
    try:
        import pkg_resources  # noqa
        # smoke: debe existir get_distribution + location
        try:
            d = pkg_resources.get_distribution("mmpose")
            _ = getattr(d, "location", None)
        except Exception:
            pass
        return {"ok": True, "already": True, "reason": "pkg_resources import ok"}
    except Exception:
        pass

    # (B) instalar setuptools en target (sin deps)
    pur_set = _purge_prefix("setuptools")
    pur_pkg = _purge_prefix("pkg_resources")  # por si quedó algo raro

    install = _pip_install_target(SPEC_SETUPTOOLS, with_deps=False, constraints_path=None)

    _activate_pydeps_for_this_process()
    importlib.invalidate_caches()

    # (C) recheck
    try:
        import pkg_resources  # noqa
        try:
            d = pkg_resources.get_distribution("mmpose")
            if not hasattr(d, "location"):
                raise RuntimeError("pkg_resources distribution has no .location")
        except Exception:
            # aunque setuptools esté, algunos entornos no exponen pkg_resources como se espera
            raise
        return {"ok": True, "already": False, "purge": {"setuptools": pur_set, "pkg_resources": pur_pkg}, "install": install, "reason": "setuptools installed -> pkg_resources ok"}
    except Exception as e:
        # (D) shim final (SIEMPRE compatible)
        shim = _write_pkg_resources_shim(PYDEPS_DIR)
        _activate_pydeps_for_this_process()
        importlib.invalidate_caches()

        # recheck shim
        try:
            import pkg_resources as pr  # noqa
            d2 = pr.get_distribution("mmpose")
            if not hasattr(d2, "location") or not d2.location:
                raise RuntimeError("shim pkg_resources missing .location")
            return {
                "ok": True,
                "already": False,
                "purge": {"setuptools": pur_set, "pkg_resources": pur_pkg},
                "install": install,
                "shim": shim,
                "reason": "shim fallback",
                "final_err": None
            }
        except Exception as e2:
            return {
                "ok": False,
                "already": False,
                "purge": {"setuptools": pur_set, "pkg_resources": pur_pkg},
                "install": install,
                "shim": shim,
                "reason": "shim fallback failed",
                "final_err": str(e2),
                "first_err": str(e),
            }


def _ensure_modules() -> Dict[str, Any]:
    pipv = _pip_ok()
    if not pipv["ok"]:
        return {"ok": False, "error": "pip not available", "pip": pipv}

    _activate_pydeps_for_this_process()

    numpy_fix = _ensure_numpy_pinned()
    if not numpy_fix.get("ok"):
        return {"ok": False, "error": "failed to pin numpy<2", "numpy_fix": numpy_fix, "pip": pipv}

    constraints = numpy_fix.get("constraints") or _write_constraints()
    hf_fix = _ensure_hf_hub_compat(constraints)

    module_plan = [
        {"name": "omegaconf", "import": "omegaconf", "spec": SPEC_OMEGACONF, "with_deps": False},
        {"name": "hydra", "import": "hydra", "spec": SPEC_HYDRA, "with_deps": False},
        {"name": "einops", "import": "einops", "spec": SPEC_EINOPS, "with_deps": False},
        {"name": "munkres", "import": "munkres", "spec": SPEC_MUNKRES, "with_deps": False},
        {"name": "mmdet", "import": "mmdet", "spec": SPEC_MMDET, "with_deps": False},
        {"name": "mmpose", "import": "mmpose", "spec": SPEC_MMPOSE, "with_deps": False},

        {"name": "transformers", "import": "transformers", "spec": SPEC_TRANSFORMERS, "with_deps": True},
        {"name": "librosa", "import": "librosa", "spec": SPEC_LIBROSA, "with_deps": True},
        {"name": "diffusers", "import": "diffusers", "spec": SPEC_DIFFUSERS, "with_deps": True},
        {"name": "xtcocotools", "import": "xtcocotools", "spec": SPEC_XTCOCO, "with_deps": True},
    ]

    ensured: Dict[str, Dict[str, Any]] = {}
    installs = []

    # 1) instalar/asegurar módulos
    for m in module_plan:
        name = m["name"]
        mod_import = m["import"]
        spec = m["spec"]
        with_deps = bool(m["with_deps"])

        ok, _ = _import_check(mod_import)
        if ok:
            ensured[name] = {"ok": True, "already": True, "pip_spec": spec, "reason": "import ok"}
            continue

        res = _pip_install_target(
            spec,
            with_deps=with_deps,
            constraints_path=constraints if with_deps else None
        )
        installs.append(res)
        ensured[name] = {"ok": None, "already": False, "pip_spec": spec, "reason": "installed -> pending recheck"}

    # 2) pkg_resources / setuptools (mmengine expects distribution.location)
    pr = _ensure_pkg_resources(constraints)
    ensured["pkg_resources"] = {"ok": pr.get("ok", False), "already": pr.get("already", False), "reason": pr.get("reason", ""), "details": pr}

    # 3) RE-CHECK FINAL
    importlib.invalidate_caches()

    final_ok = True
    for m in module_plan:
        name = m["name"]
        mod_import = m["import"]
        ok2, err2 = _import_check(mod_import)
        ensured[name]["ok"] = ok2
        if ok2:
            ensured[name]["reason"] = "import ok"
        else:
            ensured[name]["error"] = err2
            final_ok = False

    # recheck pkg_resources must work and provide .location
    pr_ok = bool(ensured["pkg_resources"].get("ok"))
    if pr_ok:
        try:
            import pkg_resources as prmod  # noqa
            d = prmod.get_distribution("mmpose")
            if not hasattr(d, "location"):
                raise RuntimeError("pkg_resources distribution has no .location")
        except Exception as e:
            ensured["pkg_resources"]["ok"] = False
            ensured["pkg_resources"]["error"] = str(e)
            pr_ok = False

    all_ok = bool(final_ok and pr_ok and hf_fix.get("ok", False))
    return {
        "ok": all_ok,
        "pip": pipv,
        "numpy_fix": numpy_fix,
        "constraints": constraints,
        "hf_fix": hf_fix,
        "ensure": ensured,
        "installs": installs,
        "pydeps_dir": PYDEPS_DIR
    }


def _import_check_in_worker() -> Dict[str, Any]:
    info: Dict[str, Any] = {
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

    # numpy
    try:
        import numpy as np  # noqa
        info["numpy"] = {"ok": True, "msg": "OK_numpy", "version": getattr(np, "__version__", None)}
    except Exception as e:
        info["numpy"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    # hf hub
    try:
        import huggingface_hub as hfh  # noqa
        info["huggingface_hub"] = {
            "ok": True,
            "msg": "OK_huggingface_hub",
            "version": getattr(hfh, "__version__", None),
            "has_cached_download": hasattr(hfh, "cached_download"),
        }
    except Exception as e:
        info["huggingface_hub"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    # pkg_resources (must provide .location)
    try:
        import pkg_resources  # noqa
        # smoke: distribution.location exists
        try:
            d = pkg_resources.get_distribution("mmpose")
            _ = getattr(d, "location", None)
        except Exception:
            pass
        info["pkg_resources"] = {"ok": True, "msg": "OK_pkg_resources"}
    except Exception as e:
        info["pkg_resources"] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    # rest
    for mod in ["munkres", "mmdet", "mmpose", "xtcocotools", "omegaconf", "transformers", "librosa", "einops", "diffusers"]:
        try:
            m = __import__(mod)
            payload: Dict[str, Any] = {"ok": True, "msg": f"OK_{mod}"}
            if mod in ("diffusers", "mmdet"):
                payload["version"] = getattr(m, "__version__", None)
            info[mod] = payload
        except Exception as e:
            info[mod] = {"ok": False, "error": str(e), "trace": _tail(traceback.format_exc())}

    info["ok"] = bool(
        info.get("core", {}).get("ok")
        and info.get("numpy", {}).get("ok")
        and str(info.get("numpy", {}).get("version", "")).startswith("1.")
        and info.get("huggingface_hub", {}).get("ok")
        and info.get("huggingface_hub", {}).get("has_cached_download", False)
        and info.get("pkg_resources", {}).get("ok")
        and info.get("munkres", {}).get("ok")
        and info.get("mmdet", {}).get("ok")
        and info.get("mmpose", {}).get("ok")
        and info.get("xtcocotools", {}).get("ok")
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
    importlib.invalidate_caches()

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
    importlib.invalidate_caches()

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

        if mode == "echo":
            return mode_echo()
        if mode in ("voice2video", "v2v"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
