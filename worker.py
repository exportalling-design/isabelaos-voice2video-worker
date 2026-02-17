# /app/worker.py
import os
import sys
import gc
import time
import base64
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, List, Tuple

# ----------------------------
# BOOT logs (para ver si arranca o muere)
# ----------------------------
print("[BOOT] starting worker...")
print("[BOOT] python:", sys.executable)
print("[BOOT] cwd:", os.getcwd())
print("[BOOT] RUNPOD_VOLUME_PATH:", os.environ.get("RUNPOD_VOLUME_PATH"))
print("[BOOT] COQUI_TOS_AGREED:", os.environ.get("COQUI_TOS_AGREED"))

# ----------------------------
# Hardening env (evita mezclar paths)
# ----------------------------
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("COQUI_TOS_AGREED", "1")  # auto-acepta CPML si Coqui lo respeta

SYS_PY = "/usr/local/bin/python3"

import runpod  # noqa: E402

# ----------------------------
# Paths
# ----------------------------
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

def _exists_file(p: str) -> bool:
    return bool(p) and os.path.isfile(p)

def _require_file(path: str, label: str):
    if not _exists_file(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _require_dir(path: str, label: str):
    if not path or not os.path.isdir(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _clean_env(extra: Dict[str, str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("COQUI_TOS_AGREED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if extra:
        env.update(extra)
    return env

def _run(cmd: List[str], cwd: str = None, env: Dict[str, str] = None, stdin_text: str = None) -> str:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env if env is not None else _clean_env(),
        input=stdin_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    out = p.stdout or ""
    if p.returncode != 0:
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{out[-12000:]}")
    return out

def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass

# ----------------------------
# IO helpers
# ----------------------------
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
    # url debe ser http(s)
    if not (url.startswith("http://") or url.startswith("https://")):
        raise RuntimeError(f"video_url/audio_url inválida: {url[:30]}...")
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())

# ----------------------------
# Voice refs
# ----------------------------
VOICES_DIR = os.environ.get("VOICES_DIR") or f"{BASE}/voices"
FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV   = os.environ.get("MALE_REF_WAV")   or f"{VOICES_DIR}/male_ref.wav"

# ----------------------------
# MuseTalk detection
# ----------------------------
def _candidate_muse_roots(base: str) -> List[str]:
    # Respeta override si existe
    ov = (os.environ.get("MUSE_ROOT") or "").strip()
    cands = []
    if ov:
        cands.append(ov)

    # tus nombres típicos
    cands += [
        f"{base}/musetalk_ok",
        f"{base}/musetalk_ok_persist",
        f"{base}/MuseTalk",
        f"{base}/musetalk",
        f"{base}/muse",
    ]

    # elimina duplicados manteniendo orden
    seen = set()
    out = []
    for p in cands:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _find_muse_root() -> str:
    for p in _candidate_muse_roots(BASE):
        if os.path.isdir(p):
            return p
    return f"{BASE}/musetalk_ok"  # fallback (para que el error sea claro)

MUSE_ROOT = _find_muse_root()

def _find_infer_script(muse_root: str) -> Tuple[str, str]:
    """
    Retorna (cwd, script_rel_or_abs)
    - Preferido: scripts/inference.py
    - Si no existe: busca cualquier archivo que parezca inference dentro del repo.
    """
    preferred = os.path.join(muse_root, "scripts", "inference.py")
    if os.path.isfile(preferred):
        return muse_root, "scripts/inference.py"

    # Buscar alternativas comunes
    common = [
        os.path.join(muse_root, "inference.py"),
        os.path.join(muse_root, "scripts", "infer.py"),
        os.path.join(muse_root, "inference", "inference.py"),
    ]
    for c in common:
        if os.path.isfile(c):
            # si está dentro del root, lo ejecutamos relativo
            if c.startswith(muse_root + "/"):
                rel = c[len(muse_root) + 1 :]
                return muse_root, rel
            return os.path.dirname(c), os.path.basename(c)

    # Búsqueda amplia (limitada)
    best = None
    for root, dirs, files in os.walk(muse_root):
        # corta directorios pesados
        dirs[:] = [d for d in dirs if d not in ("venv", ".venv", "__pycache__", ".git", "weights", "checkpoints")]
        for fn in files:
            low = fn.lower()
            if low.endswith(".py") and ("inference" in low or low == "infer.py"):
                best = os.path.join(root, fn)
                break
        if best:
            break

    if not best:
        raise RuntimeError(f"No encuentro script de inference dentro de: {muse_root}")

    if best.startswith(muse_root + "/"):
        rel = best[len(muse_root) + 1 :]
        return muse_root, rel
    return os.path.dirname(best), os.path.basename(best)

# ----------------------------
# XTTS (texto -> wav) usando /app/tts_generate.py
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

    # Por si Coqui intenta pedir y/n en algún caso
    _run(cmd, env=_clean_env(), stdin_text="y\n")

# ----------------------------
# MuseTalk run
# ----------------------------
def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    _require_dir(MUSE_ROOT, "MUSE_ROOT (MuseTalk folder)")

    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    cwd, script = _find_infer_script(MUSE_ROOT)

    # Intentamos con args típicos; si tu script usa otros flags, ajustamos después.
    cmd = [
        SYS_PY, "-u", script,
        "--inference_config", "inference_config.json",
        "--bbox_shift", "0",
        "--use_float16",
    ]

    out = _run(cmd, cwd=cwd, env=_clean_env())
    # print limitado
    print("[MuseTalk] ran:", " ".join(cmd))
    print("[MuseTalk] out_tail:", out[-800:])

    # Busca resultados
    # Algunos forks guardan en results/v15; otros en results/
    candidates = [
        os.path.join(MUSE_ROOT, "results", "v15"),
        os.path.join(MUSE_ROOT, "results"),
        os.path.join(MUSE_ROOT, "output"),
        os.path.join(MUSE_ROOT, "outputs"),
    ]

    mp4s = []
    for d in candidates:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".mp4"):
                    mp4s.append(os.path.join(d, f))

    if not mp4s:
        raise RuntimeError("MuseTalk no produjo mp4 (no encontré .mp4 en results/output)")

    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]

# ----------------------------
# Modes
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
            "MUSE_ROOT": MUSE_ROOT,
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
        }
    }

def audio_to_video(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Para probar MuseTalk SIN XTTS:
    - recibe audio_wav_b64 o audio_url
    - y video_url o video_b64
    """
    t0 = time.time()

    video_b64 = inp.get("video_b64") or inp.get("video")
    video_url = str(inp.get("video_url") or inp.get("videoUrl") or "").strip()
    if not video_b64 and not video_url:
        raise RuntimeError("Falta video_b64 o video_url")

    audio_b64 = inp.get("audio_b64") or inp.get("audio_wav_b64")
    audio_url = str(inp.get("audio_url") or "").strip()
    if not audio_b64 and not audio_url:
        raise RuntimeError("Falta audio_b64/audio_wav_b64 o audio_url")

    with tempfile.TemporaryDirectory() as td:
        in_mp4 = os.path.join(td, "in.mp4")
        in_wav = os.path.join(td, "in.wav")

        if video_url:
            _download_to_file(video_url, in_mp4)
        else:
            _b64_to_file(str(video_b64), in_mp4)

        if audio_url:
            _download_to_file(audio_url, in_wav)
        else:
            _b64_to_file(str(audio_b64), in_wav)

        out_mp4_path = _musetalk_infer(input_mp4=in_mp4, audio_wav=in_wav)

        with open(out_mp4_path, "rb") as f:
            mp4_bytes = f.read()

    return {
        "ok": True,
        "mode": "audio_to_video",
        "elapsed_s": round(time.time() - t0, 3),
        "video_b64": base64.b64encode(mp4_bytes).decode("utf-8"),
        "video_mime": "video/mp4",
        "base": BASE,
        "python": SYS_PY,
        "paths": {"MUSE_ROOT": MUSE_ROOT}
    }

# ----------------------------
# Handler
# ----------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode in ("", "echo", "debug"):
            return {
                "ok": True,
                "msg": "ECHO_OK",
                "python": SYS_PY,
                "base": BASE,
                "env": {
                    "RUNPOD_VOLUME_PATH": os.environ.get("RUNPOD_VOLUME_PATH"),
                    "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
                    "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
                },
                "checks": {
                    "muse_root_exists": os.path.isdir(MUSE_ROOT),
                    "voices_dir_exists": os.path.isdir(VOICES_DIR),
                    "female_ref_exists": _exists_file(FEMALE_REF_WAV),
                    "male_ref_exists": _exists_file(MALE_REF_WAV),
                },
                "paths": {
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                }
            }

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        if mode in ("audio_to_video", "a2v", "lipsync"):
            return audio_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|voice_to_video|audio_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})
