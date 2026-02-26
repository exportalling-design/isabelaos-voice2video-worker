# /app/worker.py
# IsabelaOS RunPod Worker — MuseTalk (voice2video) + robust PYDEPS + model path resolver + diffusers-config prune + ffmpeg join
# v31-musetalk-model-resolver + unet-config-prune-by-signature + safe-ensure (2026-02-26)

import os
import re
import gc
import json
import time
import base64
import shutil
import traceback
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import runpod

# ----------------------------
# Config / Paths
# ----------------------------
PYDEPS_DIR = os.environ.get("PYDEPS_DIR", "/runpod-volume/pydeps_py310").strip()
CONSTRAINTS = os.environ.get("PYDEPS_CONSTRAINTS", f"{PYDEPS_DIR}/_constraints.txt").strip()

MUSE_REPO = os.environ.get("MUSE_REPO", "/runpod-volume/volume_old/MuseTalk").strip()
MUSE_INFER = os.environ.get("MUSE_INFER", f"{MUSE_REPO}/scripts/inference.py").strip()
MUSE_MODELS_DIR = os.environ.get("MUSE_MODELS_DIR", f"{MUSE_REPO}/models").strip()

TAIL_LINES = int(os.environ.get("SUBPROCESS_TAIL_LINES", "260"))

WAN_COLD_EACH_JOB = os.environ.get("WAN_COLD_EACH_JOB", "1").strip() not in ("0", "false", "False")

# Cache downloads on volume (so it doesn't re-download each cold start)
TORCH_HOME = os.environ.get("TORCH_HOME", "/runpod-volume/torch_cache").strip()
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/hf_cache").strip()
os.environ["TORCH_HOME"] = TORCH_HOME
os.environ["HF_HOME"] = HF_HOME

# Env stability
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:256,garbage_collection_threshold:0.8"
)

# ----------------------------
# Helpers
# ----------------------------
def _tail(text: str, n: int = TAIL_LINES) -> str:
    try:
        lines = (text or "").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return (text or "")[-16000:]


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


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _prepend_pythonpath(paths: List[str]):
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in paths if p] + ([cur] if cur else [])
    os.environ["PYTHONPATH"] = ":".join([p for p in parts if p])


def _write_file(path: str, content: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def _pip_install(spec: str, target_dir: str, with_deps: bool = True, use_constraints: bool = True) -> Dict[str, Any]:
    python = os.environ.get("PYTHON", "/opt/conda/bin/python")
    cmd = [python, "-m", "pip", "install", "--no-cache-dir", "-q", "--target", target_dir]
    if use_constraints and Path(CONSTRAINTS).exists():
        cmd += ["-c", CONSTRAINTS]
    if not with_deps:
        cmd += ["--no-deps"]
    cmd += [spec]
    p = _run(cmd)
    return {"code": p.returncode, "spec": spec, "tail": _tail(p.stdout, 160), "with_deps": with_deps, "use_constraints": use_constraints}


def _import_ok(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _import_err(mod: str) -> Optional[str]:
    try:
        __import__(mod)
        return None
    except Exception as e:
        return str(e)


# ----------------------------
# Shims (pkg_resources + pycocotools + chumpy)
# ----------------------------
def _ensure_pkg_resources_location_shim():
    shim_path = f"{PYDEPS_DIR}/pkg_resources/__init__.py"
    if Path(shim_path).exists():
        return
    sys_site = "/opt/conda/lib/python3.10/site-packages"
    shim = f'''# auto-generated pkg_resources shim (IsabelaOS)
class _Dist:
    def __init__(self, location: str):
        self.location = location
def get_distribution(_name=None):
    return _Dist("{sys_site}")
working_set = None
'''
    _write_file(shim_path, shim)


def _ensure_pycocotools_shim_to_xtcocotools():
    pkg = Path(f"{PYDEPS_DIR}/pycocotools")
    if (pkg / "__init__.py").exists() and (pkg / "mask.py").exists():
        return
    _ensure_dir(str(pkg))
    _write_file(str(pkg / "__init__.py"), "from . import mask\n")
    _write_file(str(pkg / "mask.py"), "from xtcocotools.mask import *  # noqa\n")


def _ensure_chumpy_shim():
    # We intentionally avoid real chumpy (breaks on numpy.bool). This shim satisfies imports.
    shim_path = f"{PYDEPS_DIR}/chumpy/__init__.py"
    if Path(shim_path).exists():
        return
    shim = """# auto-generated chumpy shim (IsabelaOS)
class Ch:
    pass
__all__ = ["Ch"]
"""
    _ensure_dir(f"{PYDEPS_DIR}/chumpy")
    _write_file(shim_path, shim)


def _ensure_numpy_pin():
    # Keep numpy pinned because other deps want <2 and some libs break on 2.x
    _pip_install("numpy==1.26.4", PYDEPS_DIR, with_deps=False)

# ----------------------------
# FFmpeg ensure (system or imageio-ffmpeg fallback)
# ----------------------------
def _ensure_ffmpeg() -> Dict[str, Any]:
    import shutil as _shutil
    ff = _shutil.which("ffmpeg")
    if ff:
        return {"ok": True, "ffmpeg": ff, "source": "system"}

    # fallback: imageio-ffmpeg provides a bundled binary
    _pip_install("imageio-ffmpeg==0.4.9", PYDEPS_DIR, with_deps=True, use_constraints=False)
    try:
        import imageio_ffmpeg  # type: ignore
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        return {"ok": True, "ffmpeg": ff, "source": "imageio-ffmpeg"}
    except Exception as e:
        return {"ok": False, "error": f"ffmpeg not found and imageio-ffmpeg failed: {e}"}


def _detect_ffmpeg_pattern(frames_dir: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Detect a numeric sequence pattern like 000001.png => %06d.png and the extension.
    Returns (pattern, pad, ext)
    """
    candidates = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not candidates:
        return None, None, None

    # pick the first file that matches digits+ext
    for p in candidates[:50]:
        m = re.match(r"^(\d+)\.(png|jpg|jpeg)$", p.name.lower())
        if m:
            pad = len(m.group(1))
            ext = m.group(2)
            return f"%0{pad}d.{ext}", pad, ext

    # fallback: if names aren't numeric, ffmpeg pattern won't work reliably
    return None, None, None


def frames_to_video(frames_dir: str, out_mp4: str, fps: int = 25) -> Dict[str, Any]:
    ff = _ensure_ffmpeg()
    if not ff.get("ok"):
        return ff

    frames = Path(frames_dir)
    if not frames.exists():
        return {"ok": False, "error": f"frames_dir not found: {frames_dir}"}

    pattern, pad, ext = _detect_ffmpeg_pattern(frames)
    if not pattern:
        return {
            "ok": False,
            "error": "Could not detect numeric frame pattern (expected filenames like 000001.png).",
            "hint": f"Rename frames inside {frames_dir} to numeric sequence OR pass a proper pattern."
        }

    inp = str(frames / pattern)

    cmd = [
        ff["ffmpeg"],
        "-y",
        "-framerate", str(fps),
        "-i", inp,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        out_mp4
    ]
    p = _run(cmd)
    if p.returncode != 0:
        return {"ok": False, "error": "ffmpeg join failed", "tail": _tail(p.stdout, 200), "cmd": " ".join(cmd)}
    return {"ok": True, "out": out_mp4, "ffmpeg": ff, "pattern": pattern}


# ----------------------------
# Ensure deps in PYDEPS_DIR (SAFE: only install if missing)
# ----------------------------
def ensure_env() -> Dict[str, Any]:
    _ensure_dir(PYDEPS_DIR)
    _ensure_dir(TORCH_HOME)
    _ensure_dir(HF_HOME)

    # Activate pydeps for THIS process (important: so imports like diffusers work in worker itself)
    _prepend_pythonpath([PYDEPS_DIR, MUSE_REPO])

    installs = []
    _ensure_numpy_pin()

    # Install only if missing (do NOT reinstall everything every job)
    needed = [
        ("diffusers==0.27.2", "diffusers", False, False),
        ("transformers==4.38.2", "transformers", False, False),
        ("einops==0.7.0", "einops", False, False),
        ("hydra-core==1.3.2", "hydra", False, False),
        ("omegaconf==2.3.0", "omegaconf", False, False),
        ("munkres==1.1.4", "munkres", False, False),
        ("xtcocotools==1.13.0", "xtcocotools", False, False),
        ("shapely==2.0.3", "shapely", True, True),
        ("terminaltables==3.1.10", "terminaltables", False, False),
        ("json-tricks==3.17.3", "json_tricks", True, True),
        ("mmdet==3.3.0", "mmdet", True, True),
        ("mmpose==1.3.2", "mmpose", True, True),
        ("accelerate==0.27.2", "accelerate", True, False),
    ]

    for spec, mod, with_deps, heavy in needed:
        if not _import_ok(mod):
            installs.append(_pip_install(spec, PYDEPS_DIR, with_deps=with_deps, use_constraints=True))

    # Hard ensure accelerate importable (sometimes constraints combos block it)
    if not _import_ok("accelerate"):
        installs.append(_pip_install("accelerate==0.27.2", PYDEPS_DIR, with_deps=True, use_constraints=False))

    # Shims
    _ensure_pkg_resources_location_shim()
    _ensure_pycocotools_shim_to_xtcocotools()
    _ensure_chumpy_shim()

    # Quick import report (for debugging without breaking anything)
    imports = {
        "diffusers": {"ok": _import_ok("diffusers"), "err": _import_err("diffusers")},
        "transformers": {"ok": _import_ok("transformers"), "err": _import_err("transformers")},
        "accelerate": {"ok": _import_ok("accelerate"), "err": _import_err("accelerate")},
        "mmdet": {"ok": _import_ok("mmdet"), "err": _import_err("mmdet")},
        "mmpose": {"ok": _import_ok("mmpose"), "err": _import_err("mmpose")},
        "shapely": {"ok": _import_ok("shapely"), "err": _import_err("shapely")},
        "terminaltables": {"ok": _import_ok("terminaltables"), "err": _import_err("terminaltables")},
        "json_tricks": {"ok": _import_ok("json_tricks"), "err": _import_err("json_tricks")},
    }

    return {
        "pydeps_dir": PYDEPS_DIR,
        "constraints": CONSTRAINTS,
        "installs": installs,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "imports": imports,
        "repo": {"muse_repo": MUSE_REPO, "inference_py": MUSE_INFER, "repo_exists": Path(MUSE_REPO).exists()},
    }


# ----------------------------
# MuseTalk: model resolver + config prune (by diffusers signature)
# ----------------------------
def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _save_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _find_file(root: Path, filename: str) -> Optional[Path]:
    try:
        for p in root.rglob(filename):
            if p.is_file():
                return p
    except Exception:
        pass
    return None


def _ensure_musetalk_required_files() -> Dict[str, Any]:
    """
    Ensure MuseTalk can find the files it commonly hardcodes:
      - ./models/musetalk/config.json
      - ./models/musetalkV15/unet.pth
    We do NOT download here; we only map/link/copy existing files already on the volume.
    """
    models = Path(MUSE_MODELS_DIR)
    if not models.exists():
        return {"ok": False, "error": f"MUSE_MODELS_DIR not found: {models}"}

    actions = []

    # 1) Ensure config.json at models/musetalk/config.json
    target_cfg = models / "musetalk" / "config.json"
    if not target_cfg.exists():
        # prefer musetalkV15/config.json, else any */config.json
        v15_cfg = models / "musetalkV15" / "config.json"
        cand = v15_cfg if v15_cfg.exists() else None
        if not cand:
            any_cfgs = list(models.glob("*/config.json"))
            cand = any_cfgs[0] if any_cfgs else None
        if not cand:
            return {"ok": False, "error": f"Missing {target_cfg} and no models/*/config.json found"}
        target_cfg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cand, target_cfg)
        actions.append(f"copied {cand} -> {target_cfg}")

    # 2) Ensure unet.pth at models/musetalkV15/unet.pth (what your logs show it loads)
    expected_unet = models / "musetalkV15" / "unet.pth"
    if not expected_unet.exists():
        found = _find_file(models, "unet.pth") or _find_file(models, "unet.pt")
        if not found:
            return {
                "ok": False,
                "error": "unet.pth not found anywhere under models/",
                "hint": f"Put MuseTalk weights under {models}/musetalkV15/unet.pth (or keep them anywhere under {models} and the worker will link/copy)."
            }
        expected_unet.parent.mkdir(parents=True, exist_ok=True)

        # prefer symlink for speed; fallback to copy
        try:
            if expected_unet.exists():
                expected_unet.unlink()
            os.symlink(str(found), str(expected_unet))
            actions.append(f"symlinked {found} -> {expected_unet}")
        except Exception:
            shutil.copy2(found, expected_unet)
            actions.append(f"copied {found} -> {expected_unet}")

    return {"ok": True, "actions": actions, "config": str(target_cfg), "unet": str(expected_unet)}


def _allowed_unet_kwargs() -> List[str]:
    """
    Read diffusers.UNet2DConditionModel __init__ signature to know what keys are allowed.
    This prevents 'activation_dropout', 'activation_function', etc from crashing older diffusers.
    """
    from inspect import signature
    from diffusers import UNet2DConditionModel  # must be importable in worker process

    sig = signature(UNet2DConditionModel.__init__)
    allowed = []
    for name, p in sig.parameters.items():
        if name in ("self",):
            continue
        allowed.append(name)
    return allowed


def _prune_unet_config_for_diffusers(cfg_obj: Any, allowed_keys: List[str]) -> Tuple[Any, int]:
    """
    Prune dict keys not in allowed_keys.
    MuseTalk sometimes stores a dict of kwargs. If nested, we try common patterns.
    Returns (new_obj, removed_count).
    """
    removed = 0

    def prune_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal removed
        out = {}
        for k, v in d.items():
            if k in allowed_keys:
                out[k] = v
            else:
                removed += 1
        return out

    # Common cases:
    # - cfg is directly kwargs dict
    # - cfg has "unet_config": {...}
    # - cfg has nested {"model": {"unet": {...}}} (rare)
    if isinstance(cfg_obj, dict):
        if "unet_config" in cfg_obj and isinstance(cfg_obj["unet_config"], dict):
            cfg_obj["unet_config"] = prune_dict(cfg_obj["unet_config"])
            return cfg_obj, removed
        if "unet" in cfg_obj and isinstance(cfg_obj["unet"], dict):
            cfg_obj["unet"] = prune_dict(cfg_obj["unet"])
            return cfg_obj, removed

        # If it's already kwargs dict, prune it
        # Heuristic: if it contains typical UNet keys
        typical = {"sample_size", "in_channels", "out_channels", "layers_per_block", "block_out_channels"}
        if typical.intersection(set(cfg_obj.keys())):
            return prune_dict(cfg_obj), removed

    return cfg_obj, removed


def _musetalk_prepare_models_and_config() -> Dict[str, Any]:
    """
    1) Ensure required files exist in expected paths (no downloads).
    2) Prune config.json keys not supported by pinned diffusers.
    """
    fix_files = _ensure_musetalk_required_files()
    if not fix_files.get("ok"):
        return fix_files

    # Ensure diffusers importable in worker process (PYTHONPATH already set by ensure_env)
    allowed = _allowed_unet_kwargs()

    cfg_path = Path(fix_files["config"])
    cfg = _load_json(cfg_path)

    cfg2, removed = _prune_unet_config_for_diffusers(cfg, allowed)

    # If pruning did nothing but errors persist, we also do a direct scrub for known offenders
    # (safe even if not present)
    offenders = {"activation_dropout", "activation_function", "norm_num_groups", "mid_block_only_cross_attention"}
    if isinstance(cfg2, dict):
        # scrub top-level
        for k in list(cfg2.keys()):
            if k in offenders and k not in allowed:
                cfg2.pop(k, None)
                removed += 1
        # scrub unet_config if present
        if "unet_config" in cfg2 and isinstance(cfg2["unet_config"], dict):
            for k in list(cfg2["unet_config"].keys()):
                if k in offenders and k not in allowed:
                    cfg2["unet_config"].pop(k, None)
                    removed += 1

    if removed > 0:
        _save_json(cfg_path, cfg2)

    return {"ok": True, "actions": fix_files.get("actions", []), "config": str(cfg_path), "removed_keys": removed}


# ----------------------------
# MuseTalk subprocess runner
# ----------------------------
def _musetalk_infer_subprocess(args_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prep = _musetalk_prepare_models_and_config()
    if not prep.get("ok"):
        raise RuntimeError("MuseTalk prepare failed: " + str(prep))

    python = os.environ.get("PYTHON", "/opt/conda/bin/python")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PYDEPS_DIR}:{MUSE_REPO}:" + env.get("PYTHONPATH", "")
    env["TORCH_HOME"] = TORCH_HOME
    env["HF_HOME"] = HF_HOME

    cmd = [python, MUSE_INFER]
    if args_override:
        for k, v in args_override.items():
            cmd += [f"--{k}", str(v)]

    p = _run(cmd, env=env, cwd=MUSE_REPO)

    if p.returncode != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(p.stdout, TAIL_LINES))

    return {"ok": True, "stdout_tail": _tail(p.stdout, 220), "prepare": prep}


# ----------------------------
# Modes
# ----------------------------
def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs MuseTalk. Optional:
      - inp["musetalk_args"] : passed as flags
      - inp["join_frames"] : { "frames_dir": "...", "out": "...", "fps": 25 }
    """
    args_override = inp.get("musetalk_args")
    mus = _musetalk_infer_subprocess(args_override=args_override)

    join = inp.get("join_frames")
    joined = None
    if isinstance(join, dict):
        frames_dir = join.get("frames_dir")
        out = join.get("out", "/runpod-volume/musetalk_out.mp4")
        fps = int(join.get("fps", 25))
        if frames_dir:
            joined = frames_to_video(frames_dir, out, fps=fps)

    return {"ok": True, "musetalk": mus, "joined": joined}


def mode_health() -> Dict[str, Any]:
    models = Path(MUSE_MODELS_DIR)
    # light diagnostics without heavy scans
    listing = []
    try:
        if models.exists():
            listing = sorted([p.name for p in models.iterdir()])[:50]
    except Exception:
        pass

    return {
        "ok": True,
        "paths": {
            "PYDEPS_DIR": PYDEPS_DIR,
            "MUSE_REPO": MUSE_REPO,
            "MUSE_INFER": MUSE_INFER,
            "MUSE_MODELS_DIR": MUSE_MODELS_DIR,
            "TORCH_HOME": TORCH_HOME,
            "HF_HOME": HF_HOME,
        },
        "models_dir_exists": models.exists(),
        "models_top_listing": listing,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }


# ----------------------------
# Main handler
# ----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
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
            return {"ok": True, "mode": "health", "ensure": ensure_info, "output": mode_health()}

        return {"ok": False, "error": f"Unknown mode: {mode}", "ensure": ensure_info}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
