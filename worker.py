# /app/worker.py
# IsabelaOS RunPod Worker — MuseTalk voice2video
# v34: auto-resolve inference_config yaml (no more test_img.yaml missing) + hard ensure accelerate
# + sitecustomize UNet forced dims + ffmpeg join
# (2026-02-26)

import os
import sys
import gc
import json
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
MUSE_CONFIGS_DIR = os.environ.get("MUSE_CONFIGS_DIR", f"{MUSE_REPO}/configs").strip()

TAIL_LINES = int(os.environ.get("SUBPROCESS_TAIL_LINES", "260"))
WAN_COLD_EACH_JOB = os.environ.get("WAN_COLD_EACH_JOB", "1").strip() not in ("0", "false", "False")

# Cache downloads on volume
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

# ---- Forced dims (from your mismatch logs) ----
FORCE_IN_CHANNELS = int(os.environ.get("MUSE_FORCE_IN_CHANNELS", "8"))
FORCE_CROSS_ATT_DIM = int(os.environ.get("MUSE_FORCE_CROSS_ATT_DIM", "384"))

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
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in paths if p] + ([cur] if cur else [])
    os.environ["PYTHONPATH"] = ":".join([p for p in parts if p])

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
    return {"code": p.returncode, "spec": spec, "tail": _tail(p.stdout, 180), "with_deps": with_deps, "constraints": use_constraints}


def _import_ok(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


# ----------------------------
# Shims
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
    _write_file(str(shim_path), "# auto-generated chumpy shim (IsabelaOS)\nclass Ch:\n    pass\n__all__=['Ch']\n")


# ----------------------------
# sitecustomize monkeypatch (forces UNet config)
# ----------------------------
def _ensure_sitecustomize_monkeypatch():
    p = Path(PYDEPS_DIR) / "sitecustomize.py"
    code = f"""
# AUTO-GENERATED by IsabelaOS worker (MuseTalk hard patch)
import os

_FORCE_IN = int(os.environ.get("MUSE_FORCE_IN_CHANNELS", "{FORCE_IN_CHANNELS}"))
_FORCE_CROSS = int(os.environ.get("MUSE_FORCE_CROSS_ATT_DIM", "{FORCE_CROSS_ATT_DIM}"))

_scrub = {{
    "activation_dropout",
    "activation_function",
    "use_linear_projection",
}}

def _patch_cfg(cfg: dict):
    if not isinstance(cfg, dict):
        return cfg
    cfg["in_channels"] = _FORCE_IN
    cfg["cross_attention_dim"] = _FORCE_CROSS
    for k in list(cfg.keys()):
        if k in _scrub:
            cfg.pop(k, None)
    return cfg

try:
    from diffusers import UNet2DConditionModel
    _orig_init = UNet2DConditionModel.__init__
    def _init(self, *args, **kwargs):
        if isinstance(kwargs, dict):
            _patch_cfg(kwargs)
        return _orig_init(self, *args, **kwargs)
    UNet2DConditionModel.__init__ = _init
except Exception:
    pass
"""
    _write_file(str(p), code)


# ----------------------------
# inference_config resolver
# ----------------------------
def _resolve_inference_config_path() -> Tuple[str, Dict[str, Any]]:
    """
    Returns (path, info)
    Priority:
      1) configs/inference/test_img.yaml if exists
      2) first yaml in configs/inference/
      3) create configs/inference/_autogen.yaml fallback
    """
    cfg_root = Path(MUSE_CONFIGS_DIR)
    inf_dir = cfg_root / "inference"
    info: Dict[str, Any] = {"configs_dir": str(cfg_root), "inference_dir": str(inf_dir)}

    # 1) preferred
    preferred = inf_dir / "test_img.yaml"
    if preferred.exists():
        info["chosen"] = str(preferred)
        info["reason"] = "preferred_exists"
        return str(preferred), info

    # 2) any yaml
    cands = []
    if inf_dir.exists():
        cands += sorted(inf_dir.glob("*.yaml"))
        cands += sorted(inf_dir.glob("*.yml"))
    if cands:
        info["chosen"] = str(cands[0])
        info["reason"] = "first_found"
        info["candidates"] = [str(x) for x in cands[:10]]
        return str(cands[0]), info

    # 3) autogen minimal yaml (safe defaults)
    _ensure_dir(str(inf_dir))
    autogen = inf_dir / "_autogen.yaml"
    # Minimal schema for OmegaConf; MuseTalk expects keys it reads later.
    # We keep it generic; if MuseTalk expects additional keys, you'll see a clear KeyError next.
    autogen_text = """# auto-generated fallback inference config for MuseTalk
# If your repo has official configs, place them under configs/inference/*.yaml
device: cuda
dtype: fp16
"""
    _write_file(str(autogen), autogen_text)
    info["chosen"] = str(autogen)
    info["reason"] = "autogen_created"
    return str(autogen), info


# ----------------------------
# FFmpeg ensure (optional)
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

    ext = ".png" if pngs else ".jpg"
    inp = str(frames / f"%06d{ext}")

    cmd = [
        ff["ffmpeg"], "-y",
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
# Ensure deps in PYDEPS_DIR (no regression)
# ----------------------------
def ensure_env() -> Dict[str, Any]:
    _ensure_dir(PYDEPS_DIR)
    _ensure_dir(TORCH_HOME)
    _ensure_dir(HF_HOME)

    _prepend_paths([PYDEPS_DIR, MUSE_REPO])

    installs: List[Dict[str, Any]] = []

    if not _import_ok("numpy"):
        installs.append(_pip_install("numpy==1.26.4", PYDEPS_DIR, with_deps=False))

    if not _import_ok("diffusers"):
        installs.append(_pip_install("diffusers==0.27.2", PYDEPS_DIR, with_deps=False, use_constraints=True))

    if not _import_ok("transformers"):
        installs.append(_pip_install("transformers==4.38.2", PYDEPS_DIR, with_deps=False, use_constraints=True))

    if not _import_ok("einops"):
        installs.append(_pip_install("einops==0.7.0", PYDEPS_DIR, with_deps=False, use_constraints=True))

    # OmegaConf used by MuseTalk configs
    if not _import_ok("omegaconf"):
        installs.append(_pip_install("omegaconf==2.3.0", PYDEPS_DIR, with_deps=False, use_constraints=True))
    if not _import_ok("hydra"):
        installs.append(_pip_install("hydra-core==1.3.2", PYDEPS_DIR, with_deps=False, use_constraints=True))

    # Hard ensure accelerate importable (warning comes from transformers; we ensure it anyway)
    if not _import_ok("accelerate"):
        installs.append(_pip_install("accelerate==0.27.2", PYDEPS_DIR, with_deps=True, use_constraints=False))

    _ensure_pkg_resources_location_shim()
    _ensure_pycocotools_shim_to_xtcocotools()
    _ensure_chumpy_shim()
    _ensure_sitecustomize_monkeypatch()

    return {
        "pydeps_dir": PYDEPS_DIR,
        "constraints": CONSTRAINTS,
        "installs": installs,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "imports": {
            "diffusers": _import_ok("diffusers"),
            "accelerate": _import_ok("accelerate"),
            "transformers": _import_ok("transformers"),
            "omegaconf": _import_ok("omegaconf"),
        },
        "repo": {
            "muse_repo": MUSE_REPO,
            "inference_py": MUSE_INFER,
            "models_dir": MUSE_MODELS_DIR,
            "configs_dir": MUSE_CONFIGS_DIR,
            "repo_exists": Path(MUSE_REPO).exists(),
        },
        "force": {"in_channels": FORCE_IN_CHANNELS, "cross_attention_dim": FORCE_CROSS_ATT_DIM},
    }


# ----------------------------
# MuseTalk subprocess runner
# ----------------------------
def _musetalk_infer_subprocess(args_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    python = os.environ.get("PYTHON", "/opt/conda/bin/python")
    env = os.environ.copy()

    env["PYTHONPATH"] = f"{PYDEPS_DIR}:{MUSE_REPO}:" + env.get("PYTHONPATH", "")
    env["TORCH_HOME"] = TORCH_HOME
    env["HF_HOME"] = HF_HOME

    env["MUSE_FORCE_IN_CHANNELS"] = str(FORCE_IN_CHANNELS)
    env["MUSE_FORCE_CROSS_ATT_DIM"] = str(FORCE_CROSS_ATT_DIM)

    # Ensure inference_config exists; if user didn't supply, inject it
    cfg_path, cfg_info = _resolve_inference_config_path()

    cmd = [python, MUSE_INFER]

    injected = {}
    if args_override is None:
        args_override = {}

    # Only inject if caller did NOT specify inference_config
    if "inference_config" not in args_override and "inference-config" not in args_override:
        args_override["inference_config"] = cfg_path
        injected["inference_config"] = cfg_path

    for k, v in args_override.items():
        cmd += [f"--{k}", str(v)]

    p = _run(cmd, env=env, cwd=MUSE_REPO)

    if p.returncode != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(p.stdout, TAIL_LINES))

    return {"ok": True, "stdout_tail": _tail(p.stdout, 240), "cfg": cfg_info, "injected": injected}


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
