# /app/worker.py
# IsabelaOS RunPod Worker — MuseTalk (voice2video) + env shims + ffmpeg join
# v32-fix-scrub-write + force-musetalkV15-scrub (2026-02-26)

import os
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
    return {"code": p.returncode, "spec": spec, "tail": _tail(p.stdout, 160), "with_deps": with_deps}


def _pip_install_global(spec: str) -> Dict[str, Any]:
    python = os.environ.get("PYTHON", "/opt/conda/bin/python")
    cmd = [python, "-m", "pip", "install", "--no-cache-dir", "-q", spec]
    p = _run(cmd)
    return {"code": p.returncode, "spec": f"GLOBAL {spec}", "tail": _tail(p.stdout, 160)}


def _import_ok(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


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
    _pip_install("numpy==1.26.4", PYDEPS_DIR, with_deps=False)


# ----------------------------
# FFmpeg ensure (system or imageio-ffmpeg fallback)
# ----------------------------
def _ensure_ffmpeg() -> Dict[str, Any]:
    import shutil as _shutil
    ff = _shutil.which("ffmpeg")
    if ff:
        return {"ok": True, "ffmpeg": ff, "source": "system"}

    _pip_install("imageio-ffmpeg==0.4.9", PYDEPS_DIR, with_deps=True, use_constraints=False)
    try:
        import imageio_ffmpeg  # type: ignore
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        return {"ok": True, "ffmpeg": ff, "source": "imageio-ffmpeg"}
    except Exception as e:
        return {"ok": False, "error": f"ffmpeg not found and imageio-ffmpeg failed: {e}"}


def _detect_sequence_pattern(frames_dir: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    imgs = sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not imgs:
        return None, None, None

    img = next((p for p in imgs if p.suffix.lower() == ".png"), imgs[0])
    ext = img.suffix.lower().lstrip(".")
    stem = img.stem

    if stem.isdigit():
        width = len(stem)
        start = int(stem)
        return f"%0{width}d.{ext}", start, ext

    import re
    m = re.search(r"(\d+)$", stem)
    if m:
        digits = m.group(1)
        width = len(digits)
        start = int(digits)
        prefix = stem[:-width]
        return f"{prefix}%0{width}d.{ext}", start, ext

    return None, None, ext


def frames_to_video(frames_dir: str, out_mp4: str, fps: int = 25) -> Dict[str, Any]:
    ff = _ensure_ffmpeg()
    if not ff.get("ok"):
        return ff

    frames = Path(frames_dir)
    if not frames.exists():
        return {"ok": False, "error": f"frames_dir not found: {frames_dir}"}

    pattern, start_number, _ext = _detect_sequence_pattern(frames)

    if pattern:
        inp = str(frames / pattern)
        cmd = [ff["ffmpeg"], "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps)]
        if start_number is not None:
            cmd += ["-start_number", str(start_number)]
        cmd += ["-i", inp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), out_mp4]
        p = _run(cmd)
        if p.returncode == 0:
            return {"ok": True, "out": out_mp4, "ffmpeg": ff, "mode": "sequence", "cmd": " ".join(cmd)}

    has_png = any(frames.glob("*.png"))
    glob_pat = "*.png" if has_png else "*.jpg"
    cmd = [
        ff["ffmpeg"], "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps), "-pattern_type", "glob",
        "-i", str(frames / glob_pat),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), out_mp4
    ]
    p = _run(cmd)
    if p.returncode != 0:
        return {"ok": False, "error": "ffmpeg join failed", "tail": _tail(p.stdout, 160), "cmd": " ".join(cmd)}
    return {"ok": True, "out": out_mp4, "ffmpeg": ff, "mode": "glob", "cmd": " ".join(cmd)}


# ----------------------------
# Ensure deps in PYDEPS_DIR
# ----------------------------
def ensure_env() -> Dict[str, Any]:
    _ensure_dir(PYDEPS_DIR)
    _ensure_dir(TORCH_HOME)
    _ensure_dir(HF_HOME)

    _prepend_pythonpath([PYDEPS_DIR, MUSE_REPO])

    installs = []
    _ensure_numpy_pin()

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
        ("mmdet==3.3.0", True),
        ("mmpose==1.3.2", True),
        ("accelerate==0.27.2", True),
    ]

    for spec, with_deps in needed:
        installs.append(_pip_install(spec, PYDEPS_DIR, with_deps=with_deps))

    if not _import_ok("accelerate"):
        installs.append(_pip_install("accelerate==0.27.2", PYDEPS_DIR, with_deps=True, use_constraints=False))

    if not _import_ok("accelerate"):
        installs.append(_pip_install_global("accelerate==0.27.2"))

    _ensure_pkg_resources_location_shim()
    _ensure_pycocotools_shim_to_xtcocotools()
    _ensure_chumpy_shim()

    return {
        "pydeps_dir": PYDEPS_DIR,
        "constraints": CONSTRAINTS,
        "installs": installs,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "repo": {"muse_repo": MUSE_REPO, "inference_py": MUSE_INFER, "repo_exists": Path(MUSE_REPO).exists()},
    }


# ----------------------------
# MuseTalk config scrub (FIXED)
# ----------------------------
def _musetalk_fix_model_paths_and_scrub() -> Dict[str, Any]:
    models = Path(MUSE_MODELS_DIR)
    if not models.exists():
        return {"ok": False, "error": f"MuseTalk models dir not found: {models}"}

    actions: List[str] = []

    # Ensure models/musetalk/config.json exists (some variants expect it)
    target_dir = models / "musetalk"
    target_cfg = target_dir / "config.json"
    if not target_cfg.exists():
        v15_cfg = models / "musetalkV15" / "config.json"
        if v15_cfg.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(v15_cfg, target_cfg)
            actions.append(f"copied {v15_cfg} -> {target_cfg}")
        else:
            cands = list(models.glob("*/config.json"))
            if not cands:
                return {"ok": False, "error": f"Missing {target_cfg} and no models/*/config.json found"}
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cands[0], target_cfg)
            actions.append(f"copied {cands[0]} -> {target_cfg}")

    scrub_keys = {"activation_dropout"}

    def scrub_with_flag(obj: Any) -> Tuple[Any, bool]:
        changed = False
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k in scrub_keys:
                    obj.pop(k, None)
                    changed = True
                else:
                    obj[k], ch = scrub_with_flag(obj[k])
                    changed = changed or ch
            return obj, changed
        if isinstance(obj, list):
            new_list = []
            for x in obj:
                y, ch = scrub_with_flag(x)
                new_list.append(y)
                changed = changed or ch
            return new_list, changed
        return obj, False

    cfg_paths = list(models.glob("**/config.json"))
    if not cfg_paths:
        return {"ok": False, "error": f"No config.json found under {models}"}

    patched = 0
    for cfg_path in cfg_paths:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg2, changed = scrub_with_flag(cfg)
            if changed:
                cfg_path.write_text(json.dumps(cfg2, indent=2), encoding="utf-8")
                patched += 1
                actions.append(f"scrubbed {cfg_path} (removed activation_dropout)")
        except Exception as e:
            actions.append(f"skip {cfg_path}: {e}")

    # Force check target that your log shows:
    v15 = models / "musetalkV15" / "config.json"
    if v15.exists():
        try:
            txt = v15.read_text(encoding="utf-8")
            if "activation_dropout" in txt:
                actions.append("WARNING: activation_dropout STILL present in musetalkV15/config.json after scrub")
            else:
                actions.append("verified: musetalkV15/config.json has NO activation_dropout")
        except Exception as e:
            actions.append(f"verify v15 failed: {e}")

    actions.append(f"patched_configs={patched}/{len(cfg_paths)}")
    return {
        "ok": True,
        "actions": actions,
        "models_dir": str(models),
        "configs_found": [str(p) for p in cfg_paths],
    }


# ----------------------------
# MuseTalk subprocess runner
# ----------------------------
def _musetalk_infer_subprocess(args_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fix = _musetalk_fix_model_paths_and_scrub()
    if not fix.get("ok"):
        raise RuntimeError("MuseTalk fix failed: " + str(fix))

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

    return {"ok": True, "stdout_tail": _tail(p.stdout, 240), "fix": fix}


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
