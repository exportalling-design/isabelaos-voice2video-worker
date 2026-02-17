# /app/worker.py
import os
import gc
import time
import base64
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import runpod

# --- Hardening env ---
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("COQUI_TOS_AGREED", "1")

SYS_PY = "/usr/local/bin/python3"

def _detect_base() -> str:
    rp = (os.environ.get("RUNPOD_VOLUME_PATH") or "").strip()
    if rp and os.path.isdir(rp):
        return rp
    if os.path.isdir("/runpod-volume"):
        return "/runpod-volume"
    if os.path.isdir("/workspace"):
        return "/workspace"
    return "/"

BASE = _detect_base()

VOICES_DIR = os.environ.get("VOICES_DIR") or f"{BASE}/voices"
FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV   = os.environ.get("MALE_REF_WAV")   or f"{VOICES_DIR}/male_ref.wav"

# ----------------------------
# Helpers
# ----------------------------
def _exists_file(p: str) -> bool:
    return bool(p) and os.path.isfile(p)

def _require_file(path: str, label: str):
    if not _exists_file(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _require_dir(path: str, label: str):
    if not path or not os.path.isdir(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass

def _decode_b64(s: str) -> bytes:
    s = str(s).strip()
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()
    s = "".join(s.split()).replace("-", "+").replace("_", "/")
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad
    return base64.b64decode(s, validate=True)

def _b64_to_file(b64: str, out_path: str):
    raw = _decode_b64(b64)
    with open(out_path, "wb") as f:
        f.write(raw)

def _download_to_file(url: str, out_path: str):
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())

def _clean_env(extra: Dict[str, str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("COQUI_TOS_AGREED", "1")
    if extra:
        env.update(extra)
    return env

def _run(cmd: List[str], cwd: str = None, env: Dict[str, str] = None, stdin_text: str = None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env if env is not None else _clean_env(),
        input=stdin_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    if p.returncode != 0:
        tail = (p.stdout or "")[-12000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout or ""

# ----------------------------
# MuseTalk auto-detect
# ----------------------------
def _find_musetalk_repo(base: str) -> Tuple[Optional[str], List[str]]:
    hits = []
    candidates = [
        os.environ.get("MUSE_ROOT"),
        f"{base}/MuseTalk",
        f"{base}/volume_old/MuseTalk",
        f"{base}/volume_old/MuseTalk_repo_tmp",
    ]

    vol_old = f"{base}/volume_old"
    if os.path.isdir(vol_old):
        try:
            for name in os.listdir(vol_old):
                p = os.path.join(vol_old, name)
                if os.path.isdir(p):
                    candidates.append(p)
        except Exception:
            pass

    seen = set()
    for c in candidates:
        if not c:
            continue
        c = str(c).strip()
        if not c or c in seen:
            continue
        seen.add(c)
        inf = os.path.join(c, "scripts", "inference.py")
        if os.path.isfile(inf):
            hits.append(c)

    picked = hits[0] if hits else None
    return picked, hits

def _pick_musetalk_config(muse_root: str, base: str) -> str:
    repo_cfg = os.path.join(muse_root, "inference_config.json")
    if os.path.isfile(repo_cfg):
        return repo_cfg
    vol_cfg = os.path.join(base, "inference_config.json.")
    if os.path.isfile(vol_cfg):
        return vol_cfg
    raise RuntimeError(f"No inference_config.json found in {muse_root} or {base}")

MUSE_ROOT_PICKED, _MUSE_HITS = _find_musetalk_repo(BASE)
MUSE_CONFIG_JSON = None
if MUSE_ROOT_PICKED:
    try:
        MUSE_CONFIG_JSON = _pick_musetalk_config(MUSE_ROOT_PICKED, BASE)
    except Exception:
        MUSE_CONFIG_JSON = None

# ✅ FIX CLAVE: PYTHONPATH para que "import musetalk" funcione
def _musetalk_env() -> Dict[str, str]:
    if not MUSE_ROOT_PICKED:
        return {}
    return {"PYTHONPATH": MUSE_ROOT_PICKED}

# ----------------------------
# TTS (XTTS via Coqui TTS en imagen)
# ----------------------------
def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")

    cmd = [
        SYS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav
    ]
    _run(cmd, env=_clean_env(), stdin_text="y\n")

# ----------------------------
# MuseTalk infer
# ----------------------------
def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    if not MUSE_ROOT_PICKED:
        raise RuntimeError("MuseTalk repo not found. Run mode=scan_musetalk.")

    _require_dir(MUSE_ROOT_PICKED, "MUSE_ROOT (MuseTalk repo)")
    if not MUSE_CONFIG_JSON:
        raise RuntimeError("MuseTalk config not found (inference_config.json).")

    # Sanity: debe existir el paquete "musetalk/" en el repo root
    pkg_dir = os.path.join(MUSE_ROOT_PICKED, "musetalk")
    if not os.path.isdir(pkg_dir):
        raise RuntimeError(f"Repo picked pero no existe carpeta musetalk/: {pkg_dir}")

    inputs_dir = os.path.join(MUSE_ROOT_PICKED, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    # Asegura que el config esté dentro del repo
    cfg_name = os.path.basename(MUSE_CONFIG_JSON)
    cfg_in_repo = os.path.join(MUSE_ROOT_PICKED, cfg_name)
    if os.path.abspath(cfg_in_repo) != os.path.abspath(MUSE_CONFIG_JSON):
        _run(["bash", "-lc", f"cp -f '{MUSE_CONFIG_JSON}' '{cfg_in_repo}'"], env=_clean_env())

    cmd = [
        SYS_PY, "-u", "scripts/inference.py",
        "--inference_config", cfg_name,
        "--bbox_shift", "0",
        "--use_float16"
    ]

    # ✅ AQUÍ VA EL FIX: env incluye PYTHONPATH=repo
    env = _clean_env(_musetalk_env())
    _run(cmd, cwd=MUSE_ROOT_PICKED, env=env)

    results_dir = os.path.join(MUSE_ROOT_PICKED, "results", "v15")
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"No results dir: {results_dir}")

    mp4s = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".mp4")]
    if not mp4s:
        raise RuntimeError("MuseTalk no produjo mp4")

    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]

# ----------------------------
# Main mode
# ----------------------------
def voice_to_video(inp: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()

    text = str(inp.get("text") or "").strip()
    if not text:
        raise RuntimeError("Falta text")

    voice = str(inp.get("voice") or "female").strip().lower()
    if voice not in ("female", "male"):
        voice = "female"

    lang = str(inp.get("lang") or "es").strip().lower()
    if lang not in ("es", "en"):
        lang = "es"

    video_b64 = inp.get("video_b64") or inp.get("video")
    video_url = str(inp.get("video_url") or inp.get("videoUrl") or "").strip()
    if not video_b64 and not video_url:
        raise RuntimeError("Falta video_b64 o video_url")

    with tempfile.TemporaryDirectory() as td:
        in_mp4  = os.path.join(td, "in.mp4")
        tts_wav = os.path.join(td, "tts.wav")

        if video_url:
            _download_to_file(video_url, in_mp4)
        else:
            _b64_to_file(str(video_b64), in_mp4)

        _tts_make_wav(text=text, voice=voice, lang=lang, out_wav=tts_wav)
        out_mp4_path = _musetalk_infer(input_mp4=in_mp4, audio_wav=tts_wav)

        with open(out_mp4_path, "rb") as f:
            mp4_bytes = f.read()

    return {
        "ok": True,
        "mode": "voice_to_video",
        "elapsed_s": round(time.time() - t0, 3),
        "video_b64": base64.b64encode(mp4_bytes).decode("utf-8"),
        "video_mime": "video/mp4",
        "base": BASE,
        "python": SYS_PY,
        "paths": {
            "MUSE_ROOT": MUSE_ROOT_PICKED,
            "MUSE_CONFIG_JSON": MUSE_CONFIG_JSON,
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "PYTHONPATH_FOR_MUSETALK": MUSE_ROOT_PICKED
        }
    }

def scan_musetalk() -> Dict[str, Any]:
    picked, hits = _find_musetalk_repo(BASE)
    cfg = None
    important = []
    pkg_ok = False
    if picked:
        important.append(os.path.join(picked, "scripts", "inference.py"))
        pkg_ok = os.path.isdir(os.path.join(picked, "musetalk"))
        try:
            cfg = _pick_musetalk_config(picked, BASE)
            important.append(cfg)
        except Exception:
            pass

    return {
        "ok": True,
        "msg": "SCAN_OK",
        "base": BASE,
        "scan": {
            "muse_candidates": hits,
            "muse_root_picked": picked,
            "config_picked": cfg,
            "has_musetalk_pkg_dir": pkg_ok,
            "important_hits": important,
        }
    }

def echo() -> Dict[str, Any]:
    return {
        "ok": True,
        "msg": "ECHO_OK",
        "base": BASE,
        "python": SYS_PY,
        "env": {
            "RUNPOD_VOLUME_PATH": os.environ.get("RUNPOD_VOLUME_PATH"),
            "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
            "TTS_USE_GPU": os.environ.get("TTS_USE_GPU"),
        },
        "checks": {
            "voices_dir_exists": os.path.isdir(VOICES_DIR),
            "female_ref_exists": _exists_file(FEMALE_REF_WAV),
            "male_ref_exists": _exists_file(MALE_REF_WAV),
            "muse_root_picked": MUSE_ROOT_PICKED,
            "muse_infer_exists": bool(MUSE_ROOT_PICKED) and os.path.isfile(os.path.join(MUSE_ROOT_PICKED, "scripts", "inference.py")),
            "muse_pkg_dir_exists": bool(MUSE_ROOT_PICKED) and os.path.isdir(os.path.join(MUSE_ROOT_PICKED, "musetalk")),
            "muse_config_picked": MUSE_CONFIG_JSON,
        },
        "paths": {
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "MUSE_ROOT": MUSE_ROOT_PICKED,
            "MUSE_CONFIG_JSON": MUSE_CONFIG_JSON,
            "PYTHONPATH_FOR_MUSETALK": MUSE_ROOT_PICKED,
        }
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode in ("echo", "debug"):
            return echo()

        if mode in ("scan_musetalk", "scan"):
            return scan_musetalk()

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|scan_musetalk|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})
