# /app/worker.py
# IsabelaOS RunPod Worker — MuseTalk (voice2video) + persistent pydeps + config auto-patch from checkpoint + ffmpeg join
# v32-auto-config-from-unet + syspath-fix + smart-install (2026-02-26)

import os
import sys
import gc
import json
import shutil
import traceback
import subprocess
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, List, Set, Tuple

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

# Cache downloads on volume (avoid re-download each cold start)
TORCH_HOME = os.environ.get("TORCH_HOME", "/runpod-volume/torch_cache").strip()
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/hf_cache").strip()
os.environ["TORCH_HOME"] = TORCH_HOME
os.environ["HF_HOME"] = HF_HOME

# Env stability
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:256,garbage_collection_threshold:0.8",
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


def _prepend_paths(paths: List[str]):
    """
    CRÍTICO:
    - PYTHONPATH para subprocess
    - sys.path para imports del proceso principal
    """
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in paths if p] + ([cur] if cur else [])
    os.environ["PYTHONPATH"] = ":".join([p for p in parts if p])

    for p in reversed([p for p in paths if p]):
        if p not in sys.path:
            sys.path.insert(0, p)


def _write_file(path: str, content: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")


def _pip_install(
    spec: str, target_dir: str, with_deps: bool = True, use_constraints: bool = True
) -> Dict[str, Any]:
    python = os.environ.get("PYTHON", "/opt/conda/bin/python")
    cmd = [python, "-m", "pip", "install", "--no-cache-dir", "-q", "--target", target_dir]
    if use_constraints and Path(CONSTRAINTS).exists():
        cmd += ["-c", CONSTRAINTS]
    if not with_deps:
        cmd += ["--no-deps"]
    cmd += [spec]
    p = _run(cmd)
    return {
        "code": p.returncode,
        "spec": spec,
        "tail": _tail(p.stdout, 160),
        "with_deps": with_deps,
        "constraints": use_constraints,
    }


def _import_check(mod: str) -> Dict[str, Any]:
    try:
        __import__(mod)
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _import_ok(mod: str) -> bool:
    return _import_check(mod)["ok"]


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
    shim_dir = Path(f"{PYDEPS_DIR}/chumpy")
    shim_path = shim_dir / "__init__.py"
    if shim_path.exists():
        return
    _ensure_dir(str(shim_dir))
    shim = """# auto-generated chumpy shim (IsabelaOS)
class Ch:
    pass
__all__ = ["Ch"]
"""
    _write_file(str(shim_path), shim)


def _ensure_numpy_pin(installs: List[Dict[str, Any]]):
    if _import_ok("numpy"):
        return
    installs.append(_pip_install("numpy==1.26.4", PYDEPS_DIR, with_deps=False))


# ----------------------------
# FFmpeg ensure (system or imageio-ffmpeg fallback)
# ----------------------------
def _ensure_ffmpeg(installs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ff = shutil.which("ffmpeg")
    if ff:
        return {"ok": True, "ffmpeg": ff, "source": "system"}

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

    pngs = sorted(frames.glob("*.png"))
    jpgs = sorted(frames.glob("*.jpg")) + sorted(frames.glob("*.jpeg"))
    if not pngs and not jpgs:
        return {"ok": False, "error": f"No frames found in {frames_dir} (*.png/*.jpg)", "installs": installs}

    sample = (pngs[0].name if pngs else jpgs[0].name)
    digits = "".join([c if c.isdigit() else " " for c in sample]).split()
    if digits:
        dlen = len(digits[0])
        ext = ".png" if pngs else ".jpg"
        pat = f"%0{dlen}d{ext}"
    else:
        pat = "%06d.png" if pngs else "%06d.jpg"

    inp = str(frames / pat)
    cmd = [
        ff["ffmpeg"],
        "-y",
        "-framerate", str(fps),
        "-i", inp,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        out_mp4,
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

    _prepend_paths([PYDEPS_DIR, MUSE_REPO])

    installs: List[Dict[str, Any]] = []
    _ensure_numpy_pin(installs)

    needed = [
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
    ]

    ensure_map: Dict[str, Any] = {}
    for mod, spec, with_deps, use_constraints in needed:
        if _import_ok(mod):
            ensure_map[mod] = {"ok": True, "already": True}
            continue
        inst = _pip_install(spec, PYDEPS_DIR, with_deps=with_deps, use_constraints=use_constraints)
        installs.append(inst)
        ensure_map[mod] = {"ok": _import_ok(mod), "already": False, "installed": inst}

    # accelerate: warning only, but we hard-ensure it in pydeps (no constraints)
    if not _import_ok("accelerate"):
        installs.append(_pip_install("accelerate==0.27.2", PYDEPS_DIR, with_deps=True, use_constraints=False))
    ensure_map["accelerate"] = {"ok": _import_ok("accelerate"), "already": False}

    # Shims
    _ensure_pkg_resources_location_shim()
    _ensure_pycocotools_shim_to_xtcocotools()
    _ensure_chumpy_shim()

    return {
        "pydeps_dir": PYDEPS_DIR,
        "constraints": CONSTRAINTS,
        "installs": installs,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "sys_path_head": sys.path[:6],
        "ensure": ensure_map,
        "repo": {
            "muse_repo": MUSE_REPO,
            "inference_py": MUSE_INFER,
            "models_dir": MUSE_MODELS_DIR,
            "repo_exists": Path(MUSE_REPO).exists(),
            "has_inference_py": Path(MUSE_INFER).exists(),
        },
    }


# ----------------------------
# MuseTalk model prep + config patch from checkpoint
# ----------------------------
def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _save_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _allowed_unet_kwargs() -> Set[str]:
    from diffusers import UNet2DConditionModel  # type: ignore
    sig = inspect.signature(UNet2DConditionModel.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return allowed


def _find_any_file(models: Path, filename: str) -> Optional[Path]:
    cands = list(models.glob(f"*/{filename}"))
    if cands:
        return cands[0]
    cands = list(models.rglob(filename))
    return cands[0] if cands else None


def _ensure_musetalk_files(models: Path) -> Tuple[Path, Path, List[str]]:
    """
    Ensure:
      - models/musetalk/config.json exists (this is what MuseTalk opens)
      - models/musetalkV15/unet.pth exists (this is what MuseTalk loads)
    """
    actions: List[str] = []

    cfg_dir = models / "musetalk"
    cfg_path = cfg_dir / "config.json"
    if not cfg_path.exists():
        src = models / "musetalkV15" / "config.json"
        if not src.exists():
            src = _find_any_file(models, "config.json")
        if not src or not src.exists():
            raise FileNotFoundError(f"Missing {cfg_path} and no config.json found under {models}")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, cfg_path)
        actions.append(f"copied config {src} -> {cfg_path}")

    v15_dir = models / "musetalkV15"
    unet_path = v15_dir / "unet.pth"
    if not unet_path.exists():
        src = _find_any_file(models, "unet.pth")
        if not src or not src.exists():
            raise FileNotFoundError(f"Missing {unet_path} and no unet.pth found under {models}")
        v15_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, unet_path)
        actions.append(f"copied unet {src} -> {unet_path}")

    return cfg_path, unet_path, actions


def _infer_unet_requirements_from_checkpoint(unet_pth: Path) -> Dict[str, int]:
    """
    Reads checkpoint and extracts:
      - in_channels from conv_in.weight (shape[1])
      - cross_attention_dim from any attn2.to_k.weight (shape[1])
    """
    import torch

    raw = torch.load(str(unet_pth), map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        sd = raw["state_dict"]
    elif isinstance(raw, dict):
        sd = raw
    else:
        raise RuntimeError(f"Unexpected unet.pth format: {type(raw)}")

    in_ch = None
    cross_dim = None

    # conv_in.weight
    for k, v in sd.items():
        if k.endswith("conv_in.weight") and hasattr(v, "shape") and len(v.shape) == 4:
            in_ch = int(v.shape[1])
            break

    # attn2.to_k.weight
    for k, v in sd.items():
        if k.endswith("attn2.to_k.weight") and hasattr(v, "shape") and len(v.shape) == 2:
            cross_dim = int(v.shape[1])
            break

    if in_ch is None or cross_dim is None:
        raise RuntimeError("Could not infer in_channels/cross_attention_dim from checkpoint keys")

    return {"in_channels": in_ch, "cross_attention_dim": cross_dim}


def _patch_unet_config_values(cfg_obj: Any, in_channels: int, cross_attention_dim: int) -> Tuple[Any, List[str]]:
    """
    Recursively patch dicts that look like UNet kwargs dict.
    """
    changed: List[str] = []

    if isinstance(cfg_obj, dict):
        unet_hint_keys = {"sample_size", "in_channels", "out_channels", "layers_per_block", "block_out_channels"}
        looks_like_unet = len(unet_hint_keys.intersection(cfg_obj.keys())) >= 2

        if looks_like_unet:
            if cfg_obj.get("in_channels") != in_channels:
                cfg_obj["in_channels"] = in_channels
                changed.append(f"in_channels={in_channels}")
            if cfg_obj.get("cross_attention_dim") != cross_attention_dim:
                cfg_obj["cross_attention_dim"] = cross_attention_dim
                changed.append(f"cross_attention_dim={cross_attention_dim}")

        for k in list(cfg_obj.keys()):
            cfg_obj[k], ch2 = _patch_unet_config_values(cfg_obj[k], in_channels, cross_attention_dim)
            changed += ch2

        return cfg_obj, changed

    if isinstance(cfg_obj, list):
        out = []
        for x in cfg_obj:
            xx, ch2 = _patch_unet_config_values(x, in_channels, cross_attention_dim)
            out.append(xx)
            changed += ch2
        return out, changed

    return cfg_obj, changed


def _prune_unet_config_keys(cfg_obj: Any, allowed: Set[str], removed: List[str]) -> Any:
    """
    Remove keys not accepted by diffusers.UNet2DConditionModel.__init__ (only in UNet-like dicts).
    """
    if isinstance(cfg_obj, dict):
        unet_hint_keys = {"sample_size", "in_channels", "out_channels", "layers_per_block", "block_out_channels"}
        looks_like_unet = len(unet_hint_keys.intersection(cfg_obj.keys())) >= 2

        if looks_like_unet:
            for k in list(cfg_obj.keys()):
                if k not in allowed:
                    removed.append(k)
                    cfg_obj.pop(k, None)

        for k in list(cfg_obj.keys()):
            cfg_obj[k] = _prune_unet_config_keys(cfg_obj[k], allowed, removed)

        return cfg_obj

    if isinstance(cfg_obj, list):
        return [_prune_unet_config_keys(x, allowed, removed) for x in cfg_obj]

    return cfg_obj


def _musetalk_prepare_models_and_config() -> Dict[str, Any]:
    models = Path(MUSE_MODELS_DIR)
    if not models.exists():
        return {"ok": False, "error": f"MuseTalk models dir not found: {models}"}

    actions: List[str] = []
    try:
        cfg_path, unet_path, act = _ensure_musetalk_files(models)
        actions += act
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # Infer required values from the exact checkpoint MuseTalk will load
    try:
        req = _infer_unet_requirements_from_checkpoint(unet_path)
    except Exception as e:
        return {"ok": False, "error": f"Failed to read unet.pth to infer config: {e}", "unet_path": str(unet_path)}

    cfg = _load_json(cfg_path)

    # Patch values to match checkpoint (fixes size mismatch)
    cfg2, changed = _patch_unet_config_values(cfg, req["in_channels"], req["cross_attention_dim"])
    if changed:
        _save_json(cfg_path, cfg2)
        actions.append(f"patched config values from unet.pth: {sorted(list(set(changed)))}")

    # Prune unsupported keys for your pinned diffusers
    allowed = _allowed_unet_kwargs()
    removed: List[str] = []
    cfg3 = _prune_unet_config_keys(cfg2, allowed, removed)
    if removed:
        _save_json(cfg_path, cfg3)
        actions.append(f"pruned unsupported UNet keys: {sorted(list(set(removed)))}")

    return {
        "ok": True,
        "actions": actions,
        "config_path": str(cfg_path),
        "unet_path": str(unet_path),
        "inferred": req,
        "removed_keys": sorted(list(set(removed))),
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
