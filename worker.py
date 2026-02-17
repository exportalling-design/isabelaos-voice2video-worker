# /app/worker.py
import os
import gc
import time
import base64
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, List

import runpod

# --- hardening ---
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

# Tu config “rara” con punto final existe en el volumen:
MUSE_CONFIG_JSON = os.environ.get("MUSE_CONFIG_JSON") or f"{BASE}/inference_config.json."

# --- Helpers ---
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

def _run(cmd: list, cwd: str = None, env: Dict[str, str] = None, stdin_text: str = None):
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

# --- XTTS ---
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

    # si alguna vez vuelve a pedir y/n:
    _run(cmd, env=_clean_env(), stdin_text="y\n")

# --- MuseTalk repo discovery ---
def _pick_musetalk_repo() -> str:
    # prioridad absoluta: tu repo que SI tiene scripts/inference.py
    p1 = f"{BASE}/volume_old/MuseTalk"
    if os.path.isfile(os.path.join(p1, "scripts", "inference.py")):
        return p1

    # fallback por si lo moviste
    candidates = [
        f"{BASE}/MuseTalk",
        f"{BASE}/musetalk_ok",  # (ojo: esto es venv, usualmente NO tiene scripts/)
        f"{BASE}/volume_old/MuseTalk_repo_tmp",
    ]
    for p in candidates:
        if os.path.isfile(os.path.join(p, "scripts", "inference.py")):
            return p

    # último recurso: buscar rápido
    for root in [BASE, f"{BASE}/volume_old"]:
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                if "inference.py" in filenames and dirpath.endswith("/scripts"):
                    return os.path.dirname(dirpath)
        except Exception:
            pass

    return p1  # devuelve el esperado aunque no exista, para error claro

def _ensure_repo_config(repo_root: str) -> str:
    """
    MuseTalk scripts esperan inference_config.json en el cwd.
    Tu config buena está en /runpod-volume/inference_config.json. (con punto).
    Copiamos a <repo_root>/inference_config.json si hace falta.
    """
    _require_file(MUSE_CONFIG_JSON, "MUSE_CONFIG_JSON (inference_config.json.)")
    dst = os.path.join(repo_root, "inference_config.json")
    if not os.path.isfile(dst):
        _run(["bash", "-lc", f"cp -f '{MUSE_CONFIG_JSON}' '{dst}'"], env=_clean_env())
    return dst

def _musetalk_infer(repo_root: str, input_mp4: str, audio_wav: str) -> str:
    _require_dir(repo_root, "MUSE_ROOT (MuseTalk repo)")
    _require_file(os.path.join(repo_root, "scripts", "inference.py"), "MuseTalk scripts/inference.py")

    # inputs esperados por MuseTalk
    inputs_dir = os.path.join(repo_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    _ensure_repo_config(repo_root)

    # ✅ FIX CLAVE: PYTHONPATH = repo_root para que `import musetalk` funcione
    env = _clean_env({
        "PYTHONPATH": repo_root,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    })

    cmd = [
        SYS_PY, "-u",
        "scripts/inference.py",
        "--inference_config", "inference_config.json",
        "--bbox_shift", "0",
        "--use_float16"
    ]
    _run(cmd, cwd=repo_root, env=env)

    # dónde sale el video (esto depende de tu versión, buscamos el mp4 más nuevo)
    out_candidates: List[str] = []
    for base_dir in [
        os.path.join(repo_root, "results"),
        os.path.join(repo_root, "outputs"),
        os.path.join(repo_root, "result"),
    ]:
        if os.path.isdir(base_dir):
            for dirpath, dirnames, filenames in os.walk(base_dir):
                for fn in filenames:
                    if fn.lower().endswith(".mp4"):
                        out_candidates.append(os.path.join(dirpath, fn))

    if not out_candidates:
        raise RuntimeError("MuseTalk terminó pero no encontré ningún .mp4 en results/outputs")

    out_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return out_candidates[0]

# --- Main pipeline ---
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

    repo_root = _pick_musetalk_repo()

    with tempfile.TemporaryDirectory() as td:
        in_mp4  = os.path.join(td, "in.mp4")
        tts_wav = os.path.join(td, "tts.wav")

        if video_url:
            _download_to_file(video_url, in_mp4)
        else:
            _b64_to_file(str(video_b64), in_mp4)

        _tts_make_wav(text=text, voice=voice, lang=lang, out_wav=tts_wav)
        out_mp4_path = _musetalk_infer(repo_root=repo_root, input_mp4=in_mp4, audio_wav=tts_wav)

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
            "MUSE_ROOT_PICKED": repo_root,
            "MUSE_CONFIG_JSON": MUSE_CONFIG_JSON,
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
        }
    }

# --- Debug/scan modes ---
def _scan(inp: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = _pick_musetalk_repo()
    cfg_exists = os.path.isfile(MUSE_CONFIG_JSON)
    has_infer = os.path.isfile(os.path.join(repo_root, "scripts", "inference.py"))
    return {
        "ok": True,
        "msg": "SCAN_OK",
        "base": BASE,
        "repo_root": repo_root,
        "cfg": {"path": MUSE_CONFIG_JSON, "exists": cfg_exists},
        "repo": {
            "exists": os.path.isdir(repo_root),
            "has_scripts_inference_py": has_infer,
            "top_level": sorted(os.listdir(repo_root))[:50] if os.path.isdir(repo_root) else None
        }
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode in ("echo", "debug"):
            repo_root = _pick_musetalk_repo()
            return {
                "ok": True,
                "msg": "ECHO_OK",
                "python": SYS_PY,
                "base": BASE,
                "env": {
                    "RUNPOD_VOLUME_PATH": os.environ.get("RUNPOD_VOLUME_PATH"),
                    "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
                    "TTS_USE_GPU": os.environ.get("TTS_USE_GPU"),
                },
                "checks": {
                    "voices_dir_exists": os.path.isdir(VOICES_DIR),
                    "female_ref_exists": _exists_file(FEMALE_REF_WAV),
                    "male_ref_exists": _exists_file(MALE_REF_WAV),
                    "muse_repo_picked": repo_root,
                    "muse_repo_exists": os.path.isdir(repo_root),
                    "muse_scripts_inference_exists": os.path.isfile(os.path.join(repo_root, "scripts", "inference.py")),
                    "muse_config_exists": os.path.isfile(MUSE_CONFIG_JSON),
                },
                "paths": {
                    "VOICES_DIR": VOICES_DIR,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                    "MUSE_CONFIG_JSON": MUSE_CONFIG_JSON,
                    "MUSE_ROOT_PICKED": repo_root,
                }
            }

        if mode in ("scan", "scan_musetalk"):
            return _scan(inp)

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|scan|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})
