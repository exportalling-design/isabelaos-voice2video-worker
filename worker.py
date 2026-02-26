# /app/worker.py
# IsabelaOS RunPod Worker — MuseTalk (voice2video) + persistent pydeps + config prune + ffmpeg join
# v31-syspath-fix + smart-install + dynamic-unet-prune + model-file-ensure (2026-02-26)

import os
import sys
import gc
import json
import time
import base64
import shutil
import traceback
import subprocess
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, List, Set

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
        lines = text.splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return text[-16000:]


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
    """
    ✅ CRÍTICO:
    - Mantiene PYTHONPATH (para subprocess)
    - Y también mete rutas a sys.path (para imports del proceso principal)
    """
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in paths if p] + ([cur] if cur else [])
    os.environ["PYTHONPATH"] = ":".join([p for p in parts if p])

    # update sys.path for *this* running interpreter
    for p in reversed([p for p in paths if p]):
        if p not in sys.path:
            sys.path.insert(0, p)


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
    return {"code": p.returncode, "spec": spec, "tail": _tail(p.stdout, 160), "with_deps": with_deps, "constraints": use_constraints}


def _import_check(mod: str) -> Dict[str, Any]:
    try:
        __import__(mod)
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _import_ok(mod: str) -> bool:
    return _import_check(mod)["ok"]


# ----------------------------
# Shims (mmengine/pkg_resources + pycocotools + chumpy)
# ----------------------------
def _ensure_pkg_resources_location_shim():
    shim_path = f"{PYDEPS_DIR}/pkg_resources/__init__.py"
    if Path(shim_path).exists():
        return {"ok": True, "already": True}
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
    return {"ok": True, "already": False, "path": shim_path}


def _ensure_pycocotools_shim_to_xtcocotools():
    pkg = Path(f"{PYDEPS_DIR}/pycocotools")
    if (pkg / "__init__.py").exists() and (pkg / "mask.py").exists():
        return {"ok": True, "already": True}
    _ensure_dir(str(pkg))
    _write_file(str(pkg / "__init__.py"), "from . import mask\n")
    _write_file(str(pkg / "mask.py"), "from xtcocotools.mask import *  # noqa\n")
    return {"ok": True, "already": False, "dir": str(pkg)}


def _ensure_chumpy_shim():
    shim_dir = Path(f"{PYDEPS_DIR}/chumpy")
    shim_path = shim_dir / "__init__.py"
    if shim_path.exists():
        return {"ok": True, "already": True}

    # If real chumpy exists and breaks (numpy.bool), remove it and write shim
    if shim_dir.exists():
        try:
            # If it’s a real install folder, purge
            for p in shim_dir.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                    else:
                        shutil.rmtree(p)
                except Exception:
                    pass
        except Exception:
            pass

    shim = """# auto-generated chumpy shim (IsabelaOS)
# Purpose: satisfy mmpose optional import without numpy.bool legacy issues.
class Ch:
    pass
__all__ = ["Ch"]
"""
    _ensure_dir(str(shim_dir))
    _write_file(str(shim_path), shim)
    return {"ok": True, "already": False, "path": str(shim_path)}


def _ensure_numpy_pin(installs: List[Dict[str, Any]]):
    # keep numpy pinned to avoid incompatible legacy deps
    chk = _import_check("numpy")
    if chk["ok"]:
        return
    installs.append(_pip_install("numpy==1.26.4", PYDEPS_DIR, with_deps=False))

# ----------------------------
# FFmpeg ensure (system or imageio-ffmpeg fallback)
# ----------------------------
def _ensure_ffmpeg(installs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ff = shutil.which("ffmpeg")
    if ff:
        return {"ok": True, "ffmpeg": ff, "source": "system"}

    # fallback: imageio-ffmpeg provides a bundled binary
    if not _import_ok("imageio_ffmpeg"):
        installs.append(_pip_install("imageio-ffmpeg==0.4.9", PYDEPS_DIR, with_deps=True, use_constraints=False))

    try:
        import imageio_ffmpeg  # type: ignore
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        return {"ok": True, "ffmpeg": ff, "source": "imageio-ffmpeg"}
    except Exception as e:
        return {"ok": False, "error": f"ffmpeg not found and imageio-ffmpeg failed: {e}"}


def frames_to_video(frames_dir: str, out_mp4: str, fps: int = 25) -> Dict[str, Any]:
    installs: List[Dict[str, Any]] = []
    ff = _ensure_ffmpeg(installs)
    if not ff.get("ok"):
        return {"ok": False, "error": ff.get("error"), "installs": installs}

    frames = Path(frames_dir)
    if not frames.exists():
        return {"ok": False, "error": f"frames_dir not found: {frames_dir}", "installs": installs}

    # Decide pattern by inspecting filenames
    pngs = sorted(frames.glob("*.png"))
    jpgs = sorted(frames.glob("*.jpg")) + sorted(frames.glob("*.jpeg"))

    if not pngs and not jpgs:
        return {"ok": False, "error": f"No frames found in {frames_dir} (*.png/*.jpg)", "installs": installs}

    sample = (pngs[0].name if pngs else jpgs[0].name)

    # find digit run like 000001.png
    digits = "".join([c if c.isdigit() else " " for c in sample]).split()
    if not digits:
        # fallback pattern
        pat = "%06d.png" if pngs else "%06d.jpg"
    else:
        dlen = len(digits[0])
        ext = ".png" if pngs else ".jpg"
        pat = f"%0{dlen}d{ext}"

    inp = str(frames / pat)

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
        return {"ok": False, "error": "ffmpeg join failed", "tail": _tail(p.stdout, 160), "cmd": " ".join(cmd), "installs": installs}
    return {"ok": True, "out": out_mp4, "ffmpeg": ff, "installs": installs}


# ----------------------------
# Ensure deps in PYDEPS_DIR (SMART: only if missing)
# ----------------------------
def ensure_env() -> Dict[str, Any]:
    _ensure_dir(PYDEPS_DIR)
    _ensure_dir(TORCH_HOME)
    _ensure_dir(HF_HOME)

    # Activate pydeps for this process + subprocess
    _prepend_pythonpath([PYDEPS_DIR, MUSE_REPO])

    installs: List[Dict[str, Any]] = []

    # Numpy pin first
    _ensure_numpy_pin(installs)

    # Minimal required libs — only install if import fails
    # NOTE: do NOT remove anything; only add missing.
    needed_imports = [
        ("diffusers", "diffusers==0.27.2", False, True),
        ("transformers", "transformers==4.38.2", False, True),
        ("einops", "einops==0.7.0", False, True),
        ("hydra", "hydra-core==1.3.2", False, True),
        ("omegaconf", "omegaconf==2.3.0", False, True),
        ("munkres", "munkres==1.1.4", False, True),
        ("xtcocotools", "xtcocotools==1.13.0", False, True),
        ("shapely", "shapely==2.0.3", True, True),
        ("terminaltables", "terminaltables==3.1.10", False, True),
        ("json_tricks", "json-tricks==3.17.3", True, True),
        ("mmdet", "mmdet==3.3.0", True, True),
        ("mmpose", "mmpose==1.3.2", True, True),
        ("accelerate", "accelerate==0.27.2", True, False),  # install w/o constraints if needed
    ]

    ensure_map: Dict[str, Any] = {}
    for mod, spec, with_deps, use_constraints in needed_imports:
        chk = _import_check(mod)
        if chk["ok"]:
            ensure_map[mod] = {"ok": True, "already": True}
            continue
        inst = _pip_install(spec, PYDEPS_DIR, with_deps=with_deps, use_constraints=use_constraints)
        installs.append(inst)
        chk2 = _import_check(mod)
        ensure_map[mod] = {"ok": chk2["ok"], "already": False, "error": chk2.get("error"), "installed": inst}

    # Shims (safe)
    shim_pkg = _ensure_pkg_resources_location_shim()
    shim_coco = _ensure_pycocotools_shim_to_xtcocotools()
    shim_ch = _ensure_chumpy_shim()

    return {
        "pydeps_dir": PYDEPS_DIR,
        "constraints": CONSTRAINTS,
        "installs": installs,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "sys_path_head": sys.path[:5],
        "ensure": ensure_map,
        "shims": {
            "pkg_resources": shim_pkg,
            "pycocotools": shim_coco,
            "chumpy": shim_ch
        },
        "repo": {
            "muse_repo": MUSE_REPO,
            "inference_py": MUSE_INFER,
            "models_dir": MUSE_MODELS_DIR,
            "repo_exists": Path(MUSE_REPO).exists(),
            "has_inference_py": Path(MUSE_INFER).exists(),
        },
    }


# ----------------------------
# MuseTalk model prep + config prune (dynamic per diffusers)
# ----------------------------
def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _save_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _allowed_unet_kwargs() -> Set[str]:
    """
    Returns allowed __init__ kwargs for diffusers.UNet2DConditionModel for your pinned diffusers.
    This prevents future failures like activation_dropout / activation_function etc.
    """
    # MUST be importable in worker process (sys.path fix ensures it)
    from diffusers import UNet2DConditionModel  # type: ignore
    sig = inspect.signature(UNet2DConditionModel.__init__)
    allowed = set(sig.parameters.keys())
    # remove 'self'
    allowed.discard("self")
    return allowed


def _find_any_file(models: Path, filename: str) -> Optional[Path]:
    # search shallow first
    cands = list(models.glob(f"*/{filename}"))
    if cands:
        return cands[0]
    # deep search fallback
    cands = list(models.rglob(filename))
    return cands[0] if cands else None


def _ensure_musetalk_files(models: Path) -> Dict[str, Any]:
    """
    Ensures:
    - models/musetalk/config.json exists
    - models/musetalkV15/unet.pth exists (or we copy/symlink from another location)
    """
    actions: List[str] = []

    # Ensure config.json at models/musetalk/config.json
    target_cfg_dir = models / "musetalk"
    target_cfg = target_cfg_dir / "config.json"
    if not target_cfg.exists():
        src = models / "musetalkV15" / "config.json"
        if not src.exists():
            src = _find_any_file(models, "config.json")
        if not src or not src.exists():
            return {"ok": False, "error": f"Missing {target_cfg} and could not find any config.json under {models}"}
        target_cfg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target_cfg)
        actions.append(f"copied config: {src} -> {target_cfg}")

    # Ensure unet.pth exists at models/musetalkV15/unet.pth (because your inference prints it)
    v15_dir = models / "musetalkV15"
    v15_unet = v15_dir / "unet.pth"
    if not v15_unet.exists():
        src = _find_any_file(models, "unet.pth")
        if not src or not src.exists():
            return {"ok": False, "error": f"Missing {v15_unet} and could not find any unet.pth under {models}"}
        v15_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, v15_unet)
        actions.append(f"copied unet: {src} -> {v15_unet}")

    return {"ok": True, "actions": actions, "config": str(target_cfg), "v15_unet": str(v15_unet)}


def _prune_unet_config_for_diffusers(cfg_obj: Any, allowed: Set[str], removed: List[str]) -> Any:
    """
    Recursively remove keys that are not accepted by UNet2DConditionModel.__init__.
    Applies only to dicts that look like UNet init dicts (heuristic: many UNet keys present).
    """
    if isinstance(cfg_obj, dict):
        # If this dict looks like UNet kwargs dict: it has typical fields
        unet_hint_keys = {"sample_size", "in_channels", "out_channels", "layers_per_block", "block_out_channels"}
        looks_like_unet = len(unet_hint_keys.intersection(cfg_obj.keys())) >= 2

        if looks_like_unet:
            for k in list(cfg_obj.keys()):
                if k not in allowed:
                    removed.append(k)
                    cfg_obj.pop(k, None)

        # recurse
        for k in list(cfg_obj.keys()):
            cfg_obj[k] = _prune_unet_config_for_diffusers(cfg_obj[k], allowed, removed)

        return cfg_obj

    if isinstance(cfg_obj, list):
        return [_prune_unet_config_for_diffusers(x, allowed, removed) for x in cfg_obj]

    return cfg_obj


def _musetalk_prepare_models_and_config() -> Dict[str, Any]:
    """
    Fixes:
      - missing config.json path expected by MuseTalk
      - missing unet.pth at musetalkV15
      - prunes config keys to match your pinned diffusers (avoids activation_* etc)
    """
    models = Path(MUSE_MODELS_DIR)
    if not models.exists():
        return {"ok": False, "error": f"MuseTalk models dir not found: {models}"}

    file_fix = _ensure_musetalk_files(models)
    if not file_fix.get("ok"):
        return file_fix

    # load and prune config.json at models/musetalk/config.json (this is the one MuseTalk opens)
    cfg_path = Path(file_fix["config"])
    cfg = _load_json(cfg_path)

    allowed = _allowed_unet_kwargs()
    removed: List[str] = []
    cfg2 = _prune_unet_config_for_diffusers(cfg, allowed, removed)

    if removed:
        _save_json(cfg_path, cfg2)

    return {
        "ok": True,
        "actions": file_fix.get("actions", []),
        "config_path": str(cfg_path),
        "removed_keys": sorted(list(set(removed))),
        "allowed_count": len(allowed),
        "v15_unet": file_fix.get("v15_unet"),
    }


# ----------------------------
# MuseTalk subprocess runner
# ----------------------------
def _musetalk_infer_subprocess(args_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prep = _musetalk_prepare_models_and_config()
    if not prep.get("ok"):
        raise RuntimeError("MuseTalk prepare failed: " + str(prep))

    python = os.environ.get("PYTHON", "/opt/conda/bin/python")
    env = os.environ.copy()

    # ensure subprocess sees pydeps + repo
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

    return {"ok": True, "stdout_tail": _tail(p.stdout, 240), "prep": prep}


# ----------------------------
# Modes
# ----------------------------
def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs MuseTalk. Optional:
      - inp["musetalk_args"] : dict passed as flags
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
            return {"ok": True, "mode": "health", "ensure": ensure_info}

        return {"ok": False, "error": f"Unknown mode: {mode}", "ensure": ensure_info}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
