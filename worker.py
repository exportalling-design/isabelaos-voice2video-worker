# /app/worker.py
# IsabelaOS RunPod Worker — MuseTalk voice2video
# v36: arg-whitelist (drop unsupported like --task_id) + audio_path autogen + UNet patch
# + accelerate best-effort (disk full won't fail) + optional ffmpeg join
# (2026-02-26)

import os
import sys
import gc
import shutil
import traceback
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

TORCH_HOME = os.environ.get("TORCH_HOME", "/runpod-volume/torch_cache").strip()
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/hf_cache").strip()
os.environ["TORCH_HOME"] = TORCH_HOME
os.environ["HF_HOME"] = HF_HOME

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:256,garbage_collection_threshold:0.8",
)

# Forced dims (from mismatch logs)
FORCE_IN_CHANNELS = int(os.environ.get("MUSE_FORCE_IN_CHANNELS", "8"))
FORCE_CROSS_ATT_DIM = int(os.environ.get("MUSE_FORCE_CROSS_ATT_DIM", "384"))

# MuseTalk inference.py accepted args (from your usage dump)
ALLOWED_FLAGS = {
    "ffmpeg_path",
    "gpu_id",
    "vae_type",
    "unet_config",
    "unet_model_path",
    "whisper_dir",
    "inference_config",
    "bbox_shift",
    "result_dir",
    "extra_margin",
    "fps",
    "audio_padding_length_left",
    "audio_padding_length_right",
    "batch_size",
    "output_vid_name",
    "use_saved_coord",
    "saved_coord",
    "use_float16",
    "parsing_mode",
    "left_cheek_width",
    "right_cheek_width",
    "version",
}

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
    return {
        "code": p.returncode,
        "spec": spec,
        "tail": _tail(p.stdout, 220),
        "with_deps": with_deps,
        "constraints": use_constraints,
    }


def _import_ok(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


# ----------------------------
# Shims + monkeypatch
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
# YAML helpers (no PyYAML dependency)
# ----------------------------
def _yaml_contains_audio_path(p: Path) -> bool:
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
        return "audio_path" in t
    except Exception:
        return False


def _resolve_inference_config_path(inp: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Prefer config that contains 'audio_path'. Avoid realtime.yaml if it doesn't.
    If none, autogen _autogen.yaml with audio_path+video_path.
    """
    info: Dict[str, Any] = {}

    user_args = inp.get("musetalk_args") or {}
    if not isinstance(user_args, dict):
        user_args = {}

    user_cfg = user_args.get("inference_config") or user_args.get("inference-config")
    if user_cfg:
        p = Path(str(user_cfg))
        if p.exists():
            info["chosen"] = str(p)
            info["reason"] = "user_provided"
            return str(p), info

    inf_dir = Path(MUSE_CONFIGS_DIR) / "inference"
    info["inference_dir"] = str(inf_dir)

    preferred = inf_dir / "test.yaml"
    if preferred.exists() and _yaml_contains_audio_path(preferred):
        info["chosen"] = str(preferred)
        info["reason"] = "preferred_test_yaml"
        return str(preferred), info

    cands: List[Path] = []
    if inf_dir.exists():
        cands += sorted(inf_dir.glob("*.yaml"))
        cands += sorted(inf_dir.glob("*.yml"))

    audio_cands = [c for c in cands if _yaml_contains_audio_path(c)]
    if audio_cands:
        info["chosen"] = str(audio_cands[0])
        info["reason"] = "first_with_audio_path"
        info["candidates"] = [str(x) for x in audio_cands[:10]]
        return str(audio_cands[0]), info

    # AUTOGEN from input
    video_path = inp.get("video_path") or user_args.get("video_path")
    audio_path = inp.get("audio_path") or user_args.get("audio_path")
    bbox_shift = inp.get("bbox_shift") or user_args.get("bbox_shift") or 5

    if not video_path or not audio_path:
        video_path = video_path or "data/video/yongen.mp4"
        audio_path = audio_path or "data/audio/yongen.wav"
        info["note"] = "video_path/audio_path not provided; autogen uses placeholders"

    _ensure_dir(str(inf_dir))
    autogen = inf_dir / "_autogen.yaml"
    content = f"""# auto-generated MuseTalk inference config (IsabelaOS)
avator_1:
  preparation: true
  bbox_shift: {int(bbox_shift)}
  video_path: "{video_path}"
  audio_path: "{audio_path}"
"""
    _write_file(str(autogen), content)
    info["chosen"] = str(autogen)
    info["reason"] = "autogen_with_audio_path"
    return str(autogen), info


# ----------------------------
# Arg filtering (drop unsupported like task_id)
# ----------------------------
def _sanitize_musetalk_args(args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    kept: Dict[str, Any] = {}
    dropped: Dict[str, Any] = {}

    for k, v in (args or {}).items():
        k_norm = str(k).lstrip("-").replace("-", "_").strip()
        if k_norm in ALLOWED_FLAGS:
            kept[k_norm] = v
        else:
            dropped[k_norm] = v

    return kept, {
        "dropped": dropped,
        "kept_keys": sorted(list(kept.keys())),
        "allowed_keys": sorted(list(ALLOWED_FLAGS)),
    }


# ----------------------------
# FFmpeg join (optional)
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
# Ensure deps
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

    if not _import_ok("omegaconf"):
        installs.append(_pip_install("omegaconf==2.3.0", PYDEPS_DIR, with_deps=False, use_constraints=True))
    if not _import_ok("hydra"):
        installs.append(_pip_install("hydra-core==1.3.2", PYDEPS_DIR, with_deps=False, use_constraints=True))

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
# MuseTalk subprocess
# ----------------------------
def _musetalk_infer_subprocess(inp: Dict[str, Any]) -> Dict[str, Any]:
    python = os.environ.get("PYTHON", "/opt/conda/bin/python")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PYDEPS_DIR}:{MUSE_REPO}:" + env.get("PYTHONPATH", "")
    env["TORCH_HOME"] = TORCH_HOME
    env["HF_HOME"] = HF_HOME
    env["MUSE_FORCE_IN_CHANNELS"] = str(FORCE_IN_CHANNELS)
    env["MUSE_FORCE_CROSS_ATT_DIM"] = str(FORCE_CROSS_ATT_DIM)

    raw_args = inp.get("musetalk_args") or {}
    if not isinstance(raw_args, dict):
        raw_args = {}

    args_override, args_info = _sanitize_musetalk_args(raw_args)

    cfg_path, cfg_info = _resolve_inference_config_path(inp)
    if "inference_config" not in args_override:
        args_override["inference_config"] = cfg_path

    cmd = [python, MUSE_INFER]
    for k, v in args_override.items():
        if isinstance(v, bool):
            if v:
                cmd += [f"--{k}"]
            continue
        cmd += [f"--{k}", str(v)]

    p = _run(cmd, env=env, cwd=MUSE_REPO)
    if p.returncode != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(p.stdout, TAIL_LINES))

    return {
        "ok": True,
        "stdout_tail": _tail(p.stdout, 260),
        "cfg": cfg_info,
        "args_used": args_override,
        "args_filter": args_info,
    }


# ----------------------------
# Modes
# ----------------------------
def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    mus = _musetalk_infer_subprocess(inp)

    joined = None
    join = inp.get("join_frames")
    if isinstance(join, dict):
        frames_dir = join.get("frames_dir")
        if frames_dir:
            out = join.get("out", "/runpod-volume/musetalk_out.mp4")
            fps = int(join.get("fps", 25))
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
