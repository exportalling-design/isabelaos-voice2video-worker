# /app/worker.py
# IsabelaOS RunPod Worker — WAN + MuseTalk (voice2video)
# v29-musetalk-cwd-config-fix + accelerate + shims (2026-02-26)

import os
import io
import re
import gc
import json
import time
import base64
import shutil
import traceback
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import runpod

# ----------------------------
# Config / Paths
# ----------------------------
PYDEPS_DIR = os.environ.get("PYDEPS_DIR", "/runpod-volume/pydeps_py310").strip()
CONSTRAINTS = os.environ.get("PYDEPS_CONSTRAINTS", f"{PYDEPS_DIR}/_constraints.txt").strip()

# MuseTalk repo + script
MUSE_REPO = os.environ.get("MUSE_REPO", "/runpod-volume/volume_old/MuseTalk").strip()
MUSE_INFER = os.environ.get("MUSE_INFER", f"{MUSE_REPO}/scripts/inference.py").strip()

# Where MuseTalk expects ./models/...
# (must be relative to cwd=MUSE_REPO)
MUSE_MODELS_DIR = os.environ.get("MUSE_MODELS_DIR", f"{MUSE_REPO}/models").strip()

# Controls tail length when subprocess fails
TAIL_LINES = int(os.environ.get("SUBPROCESS_TAIL_LINES", "220"))

# Optional: if you want to force low_cpu_mem to work, keep accelerate installed
# WAN config etc. (left for your pipeline)
WAN_COLD_EACH_JOB = os.environ.get("WAN_COLD_EACH_JOB", "1").strip() not in ("0", "false", "False")

# Keep env stable
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:256,garbage_collection_threshold:0.8"
)

# ----------------------------
# Helpers
# ----------------------------
def _log(msg: str):
    print(msg, flush=True)

def _tail(text: str, n: int = TAIL_LINES) -> str:
    try:
        lines = text.splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return text[-12000:]

def _run(cmd, env=None, cwd=None, timeout=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=cwd,
        timeout=timeout,
    )

def _pip_install(spec: str, target_dir: str, with_deps: bool = True) -> Dict[str, Any]:
    """
    Install into a target directory (pydeps) so we don't touch system site-packages.
    """
    python = os.environ.get("PYTHON", "/opt/conda/bin/python")
    cmd = [python, "-m", "pip", "install", "--no-cache-dir", "-q", "--target", target_dir]
    if Path(CONSTRAINTS).exists():
        cmd += ["-c", CONSTRAINTS]
    if not with_deps:
        cmd += ["--no-deps"]
    cmd += [spec]
    p = _run(cmd)
    return {"code": p.returncode, "spec": spec, "tail": _tail(p.stdout, 120), "with_deps": with_deps}

def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _prepend_pythonpath(paths):
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in paths if p] + ([cur] if cur else [])
    os.environ["PYTHONPATH"] = ":".join([p for p in parts if p])

def _write_file(path: str, content: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")

# ----------------------------
# Shims / Fixes for MMEngine + numpy
# ----------------------------
def _ensure_numpy_pin():
    # Many mm* stacks are happier with numpy 1.26.x in py310
    _pip_install("numpy==1.26.4", PYDEPS_DIR, with_deps=False)

def _ensure_pkg_resources_location_shim():
    """
    Some mmengine versions call pkg.location; on some environments the distribution object
    isn't what mmengine expects. We provide a shim pkg_resources that exposes `.location`.
    """
    shim_path = f"{PYDEPS_DIR}/pkg_resources/__init__.py"
    if Path(shim_path).exists():
        return

    # We set location to system site-packages, which is what mmengine wants for path resolution.
    sys_site = "/opt/conda/lib/python3.10/site-packages"
    shim = f'''# auto-generated pkg_resources shim (IsabelaOS)
# Provides minimal API + a distribution object with .location

import os as _os

class _Dist:
    def __init__(self, location: str):
        self.location = location

def get_distribution(_name=None):
    return _Dist("{sys_site}")

# Some libs check for working_set
working_set = None
'''
    _write_file(shim_path, shim)

def _ensure_pycocotools_shim_to_xtcocotools():
    """
    mmdet sometimes imports pycocotools.mask. If only xtcocotools is present,
    provide a pycocotools package that re-exports it.
    """
    pkg = Path(f"{PYDEPS_DIR}/pycocotools")
    if (pkg / "__init__.py").exists() and (pkg / "mask.py").exists():
        return

    # create shim package
    _ensure_dir(str(pkg))
    _write_file(str(pkg / "__init__.py"), "from . import mask\n")
    _write_file(
        str(pkg / "mask.py"),
        "from xtcocotools.mask import *  # noqa\n"
    )

def _ensure_chumpy_shim():
    """
    chumpy is deprecated and breaks on numpy>=1.24 due to numpy.bool removal.
    mmpose lists it, but MuseTalk generally doesn't need full chumpy runtime.
    Provide a tiny shim so imports don't crash.
    """
    shim_path = f"{PYDEPS_DIR}/chumpy/__init__.py"
    if Path(shim_path).exists():
        return

    shim = """# auto-generated chumpy shim (IsabelaOS)
# Prevents numpy.bool import crash. Only minimal placeholders.

class Ch:
    pass

__all__ = ["Ch"]
"""
    _ensure_dir(f"{PYDEPS_DIR}/chumpy")
    _write_file(shim_path, shim)

# ----------------------------
# Ensure dependencies into PYDEPS_DIR
# ----------------------------
def ensure_env() -> Dict[str, Any]:
    """
    Ensures required Python deps exist in PYDEPS_DIR and activates PYTHONPATH.
    """
    _ensure_dir(PYDEPS_DIR)

    # Activate pydeps
    _prepend_pythonpath([PYDEPS_DIR, MUSE_REPO])

    installs = []
    # Keep numpy pinned
    _ensure_numpy_pin()

    # Core deps we rely on
    needed = [
        ("diffusers==0.27.2", False),
        ("transformers==4.38.2", False),
        ("einops==0.7.0", False),
        ("hydra-core==1.3.2", False),
        ("omegaconf==2.3.0", False),
        ("munkres==1.1.4", False),
        ("xtcocotools==1.13.0", False),
        ("shapely==2.0.3", True),
        ("terminaltables==3.1.10", False),
        ("json-tricks==3.17.3", True),
        # accelerate to satisfy warning + faster model loading
        ("accelerate==0.27.2", True),
        # mmdet/mmpose are in your log as already importable from PYDEPS
        ("mmdet==3.3.0", True),
        ("mmpose==1.3.2", True),
    ]

    # Best-effort installs; if already there, pip will be quick.
    for spec, with_deps in needed:
        r = _pip_install(spec, PYDEPS_DIR, with_deps=with_deps)
        installs.append(r)

    # Shims last
    _ensure_pkg_resources_location_shim()
    _ensure_pycocotools_shim_to_xtcocotools()
    _ensure_chumpy_shim()

    return {
        "pydeps_dir": PYDEPS_DIR,
        "constraints": CONSTRAINTS,
        "installs": installs,
        "pythopath_effective_preview": os.environ.get("PYTHONPATH", ""),
        "repo": {"muse_repo": MUSE_REPO, "inference_py": MUSE_INFER, "repo_exists": Path(MUSE_REPO).exists()},
    }

# ----------------------------
# MuseTalk model path auto-fix
# ----------------------------
def _musetalk_fix_model_paths() -> Dict[str, Any]:
    """
    Fix the exact error you got:
      FileNotFoundError: './models/musetalk/config.json'

    Strategy:
      - Ensure cwd is MUSE_REPO when running inference.py
      - Ensure models/musetalk/config.json exists:
           If models/musetalkV15/config.json exists -> copy it.
           Else try to find any models/**/config.json -> copy first match.
    """
    info = {"ok": True, "actions": []}
    models = Path(MUSE_MODELS_DIR)
    if not models.exists():
        return {"ok": False, "error": f"MuseTalk models dir not found: {models}"}

    target_dir = models / "musetalk"
    target_cfg = target_dir / "config.json"
    if target_cfg.exists():
        info["actions"].append(f"config ok: {target_cfg}")
        return info

    # Preferred source
    v15_cfg = models / "musetalkV15" / "config.json"
    if v15_cfg.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v15_cfg, target_cfg)
        info["actions"].append(f"copied {v15_cfg} -> {target_cfg}")
        return info

    # Fallback: search any config.json under models/*
    candidates = list(models.glob("*/config.json"))
    if candidates:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], target_cfg)
        info["actions"].append(f"copied {candidates[0]} -> {target_cfg}")
        return info

    return {
        "ok": False,
        "error": f"Missing {target_cfg}. No source config.json found under {models}. "
                 f"Expected one at models/musetalkV15/config.json or models/*/config.json"
                 }

# ----------------------------
# MuseTalk subprocess runner
# ----------------------------
def _musetalk_infer_subprocess(args_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs MuseTalk inference.py in a subprocess with cwd=MUSE_REPO.
    """
    # Fix relative paths + config.json
    fix = _musetalk_fix_model_paths()
    if not fix.get("ok"):
        raise RuntimeError("MuseTalk model path fix failed: " + str(fix))

    python = os.environ.get("PYTHON", "/opt/conda/bin/python")

    # IMPORTANT: run with cwd=MUSE_REPO so './models/...' resolves correctly
    env = os.environ.copy()
    # Ensure pydeps and repo are in path for the subprocess too
    env["PYTHONPATH"] = f"{PYDEPS_DIR}:{MUSE_REPO}:" + env.get("PYTHONPATH", "")

    # You may already build a config file in your pipeline.
    # If you have an inference config file, set it here:
    # Example:
    #   env["MUSE_CONFIG"] = "/runpod-volume/....json"
    #
    # Otherwise, MuseTalk's inference.py likely uses argparse; we run it with no extra flags
    # and rely on your existing defaults inside the repo.
    cmd = [python, MUSE_INFER]
    if args_override:
        # If your inference.py supports flags, map them here.
        # (Kept generic; safe: --key value)
        for k, v in args_override.items():
            cmd += [f"--{k}", str(v)]

    p = _run(cmd, env=env, cwd=MUSE_REPO)

    if p.returncode != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(p.stdout, TAIL_LINES))

    return {"ok": True, "stdout_tail": _tail(p.stdout, 200), "fix": fix}

# ----------------------------
# Modes
# ----------------------------
def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your pipeline should:
      - prepare input video + audio paths
      - call MuseTalk to generate lip-synced frames/video
      - merge results and upload to Supabase etc.

    Here we only run the MuseTalk step (as your current flow).
    """
    # If you want to pass args to inference.py, put them in inp["musetalk_args"].
    args_override = inp.get("musetalk_args")
    info = _musetalk_infer_subprocess(args_override=args_override)
    return {"ok": True, "musetalk": info}

# ----------------------------
# Main RunPod handler
# ----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Ensure deps (pydeps) every run (fast if already done)
        ensure_info = ensure_env()

        inp = event.get("input") or {}
        mode = (inp.get("mode") or "voice2video").strip()

        if WAN_COLD_EACH_JOB:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        if mode == "voice2video":
            out = mode_voice2video(inp)
            return {"ok": True, "mode": mode, "ensure": ensure_info, "output": out}

        if mode == "health":
            return {"ok": True, "mode": "health", "ensure": ensure_info}

        return {"ok": False, "error": f"Unknown mode: {mode}", "ensure": ensure_info}

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }

runpod.serverless.start({"handler": handler})
