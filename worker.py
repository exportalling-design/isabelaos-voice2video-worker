import os
import sys
import time
import re
import traceback
import subprocess
from typing import Any, Dict, Optional, Tuple, List

import runpod

WORKER_VERSION_TAG = "v7-autoinstall-missing-modules-to-pydeps-2026-02-23"

RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "30"))

PY_CONTAINER = os.environ.get("PY_CONTAINER", sys.executable)

MUSE_REPO = os.environ.get(
    "MUSE_REPO",
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk")
)

PYDEPS_DIR = os.environ.get("PYDEPS_DIR", os.path.join(RUNPOD_VOLUME_PATH, "pydeps_py310"))

MMPPOSE_SPECS_ENV = os.environ.get("MMPPOSE_SPECS", "").strip()
if MMPPOSE_SPECS_ENV:
    MMPPOSE_SPECS = [s.strip() for s in MMPPOSE_SPECS_ENV.split(",") if s.strip()]
else:
    MMPPOSE_SPECS = ["mmpose==1.3.2", "mmpose==1.2.0", "mmpose==1.1.0", "mmpose"]

# Mapeos seguros: "import name" -> "pip package spec"
IMPORT_TO_PIP = {
    "omegaconf": "omegaconf==2.3.0",
    "hydra": "hydra-core==1.3.2",
    "hydra_core": "hydra-core==1.3.2",
    "yaml": "pyyaml==6.0.1",
    "tqdm": "tqdm==4.66.1",
    "einops": "einops==0.7.0",
    "scipy": "scipy==1.11.4",
    "skimage": "scikit-image==0.22.0",
    "PIL": "pillow==10.2.0",
    "moviepy": "moviepy==1.0.3",
    "soundfile": "soundfile==0.12.1",
    "librosa": "librosa==0.10.1",
}

MISSING_RE = re.compile(r"ModuleNotFoundError:\s+No module named '([^']+)'")

def _now() -> float:
    return time.time()

def _tail(s: str, n: int = 2500) -> str:
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

def _container_versions() -> Dict[str, Any]:
    env = _clean_env(None)
    code, out = _run(
        [
            PY_CONTAINER,
            "-c",
            "import sys; print('PY',sys.executable); "
            "import mmcv, mmengine; "
            "print('mmcv',mmcv.__version__); print('mmengine',mmengine.__version__)"
        ],
        env=env,
        timeout=SCAN_TIMEOUT_SEC,
    )
    return {"ok": code == 0, "code": code, "out_tail": _tail(out)}

def _pip_install_to_pydeps(spec: str) -> Dict[str, Any]:
    os.makedirs(PYDEPS_DIR, exist_ok=True)
    env = _clean_env(None)

    c0, o0 = _run([PY_CONTAINER, "-m", "pip", "--version"], env=env, timeout=SCAN_TIMEOUT_SEC)
    if c0 != 0:
        return {"ok": False, "error": "pip not available", "pip_tail": _tail(o0), "spec": spec}

    cmd = [
        PY_CONTAINER, "-m", "pip", "install",
        "--no-cache-dir",
        "--target", PYDEPS_DIR,
        "--no-build-isolation",
        spec
    ]
    c1, o1 = _run(cmd, env=env, timeout=HARD_TIMEOUT_SEC)
    return {"ok": c1 == 0, "code": c1, "spec": spec, "tail": _tail(o1)}

def _ensure_mmpose_available() -> Dict[str, Any]:
    _activate_pydeps_for_this_process()
    try:
        import mmpose  # noqa
        return {"ok": True, "already": True, "reason": "mmpose import ok"}
    except Exception:
        pass

    tried = []
    for spec in MMPPOSE_SPECS:
        r = _pip_install_to_pydeps(spec)
        tried.append(r)
        if r.get("ok"):
            _activate_pydeps_for_this_process()
            try:
                import mmpose  # noqa
                return {"ok": True, "already": False, "picked": spec, "tried": tried}
            except Exception as e:
                tried.append({"ok": False, "error": f"installed but still cannot import: {e}"})
    return {"ok": False, "tried": tried}

def _import_check_in_worker() -> Dict[str, Any]:
    info = {"py_container": PY_CONTAINER, "sys_executable": sys.executable, "pydeps_dir": PYDEPS_DIR}
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

def _find_missing_module(out: str) -> Optional[str]:
    m = MISSING_RE.search(out or "")
    if not m:
        return None
    return m.group(1).strip()

def _pip_spec_for_missing(mod: str) -> str:
    # algunos imports vienen como "package.submodule"
    root = mod.split(".")[0]
    return IMPORT_TO_PIP.get(mod) or IMPORT_TO_PIP.get(root) or root

def _musetalk_infer_once() -> Tuple[int, str]:
    env = _clean_env(MUSE_REPO)
    cmd = [PY_CONTAINER, "-u", "scripts/inference.py", "--inference_config", "inference_config.json"]
    return _run(cmd, cwd=MUSE_REPO, env=env, timeout=HARD_TIMEOUT_SEC)

def _musetalk_infer_with_autofix() -> Dict[str, Any]:
    if not os.path.isdir(MUSE_REPO):
        raise RuntimeError(f"MuseTalk repo not found: {MUSE_REPO}")
    if not os.path.isfile(os.path.join(MUSE_REPO, "scripts", "inference.py")):
        raise RuntimeError("MuseTalk scripts/inference.py not found in repo")

    # 1er intento
    code, out = _musetalk_infer_once()
    if code == 0:
        return {"ok": True, "attempts": 1, "log_tail": _tail(out), "output_mp4_guess": _guess_latest_mp4()}

    missing = _find_missing_module(out)
    if not missing:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

    # instalar solo el missing en pydeps
    spec = _pip_spec_for_missing(missing)
    inst = _pip_install_to_pydeps(spec)

    # reintento 1 vez
    _activate_pydeps_for_this_process()
    code2, out2 = _musetalk_infer_once()
    if code2 == 0:
        return {
            "ok": True,
            "attempts": 2,
            "fixed_missing": missing,
            "installed_spec": spec,
            "install": inst,
            "log_tail": _tail(out2),
            "output_mp4_guess": _guess_latest_mp4(),
        }

    raise RuntimeError(
        "MuseTalk inference failed after autofix\n"
        f"missing={missing} spec={spec}\n"
        + _tail(out2)
    )

def _guess_latest_mp4() -> Optional[str]:
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
    return newest

def mode_scan() -> Dict[str, Any]:
    return {
        "ok": True,
        "msg": "SCAN_OK",
        "worker_version": WORKER_VERSION_TAG,
        "repo": _repo_check(),
        "py_container": PY_CONTAINER,
        "sys_executable": sys.executable,
        "pydeps_dir": PYDEPS_DIR,
        "versions": _container_versions(),
    }

def mode_echo() -> Dict[str, Any]:
    repo = _repo_check()
    versions = _container_versions()

    _activate_pydeps_for_this_process()
    mmp = _ensure_mmpose_available()
    chk = _import_check_in_worker()

    return {
        "ok": True,
        "msg": "ECHO_OK",
        "worker_version": WORKER_VERSION_TAG,
        "repo": repo,
        "versions": versions,
        "ensure_mmpose": mmp,
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
    _activate_pydeps_for_this_process()

    mmp = _ensure_mmpose_available()
    if not mmp.get("ok"):
        raise RuntimeError("mmpose not available\n" + str(mmp))

    chk = _import_check_in_worker()
    if not chk.get("ok"):
        raise RuntimeError("Deps import check failed\n" + str(chk))

    # ✅ aquí corre MuseTalk y si falta algo tipo omegaconf lo instala a pydeps y reintenta 1 vez
    info = _musetalk_infer_with_autofix()

    elapsed = int((_now() - start) * 1000)
    return {
        "ok": True,
        "msg": "VOICE2VIDEO_OK",
        "worker_version": WORKER_VERSION_TAG,
        "execution_ms": elapsed,
        "repo": repo,
        "versions": versions,
        "ensure_mmpose": mmp,
        "imports": chk,
        "python_used": PY_CONTAINER,
        "output_mp4_guess": info.get("output_mp4_guess"),
        "attempts": info.get("attempts"),
        "fixed_missing": info.get("fixed_missing"),
        "installed_spec": info.get("installed_spec"),
        "install": info.get("install"),
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
