# /app/worker.py
import os
import gc
import time
import base64
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, Tuple, List

import runpod

# Limpieza de contaminación global
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
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if extra:
        env.update(extra)
    return env

def _run(cmd: list, cwd: str = None, env: Dict[str, str] = None, stdin_text: str = None) -> str:
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

# ---------- MuseTalk detection ----------
def _find_musetalk_repo_candidates() -> List[str]:
    cands = []
    # Tu caso real
    p1 = f"{BASE}/volume_old/MuseTalk"
    if os.path.isdir(p1):
        cands.append(p1)
    # posibles
    p2 = f"{BASE}/MuseTalk"
    if os.path.isdir(p2):
        cands.append(p2)
    p3 = f"{BASE}/musetalk_ok"
    # ojo: musetalk_ok parece ser un venv, NO repo. Lo metemos al final por si acaso.
    if os.path.isdir(p3):
        cands.append(p3)
    return cands

def _pick_musetalk_repo() -> Tuple[str, str]:
    """
    Retorna (repo_root, config_path_abs)
    """
    # config: tu volumen tiene inference_config.json. con punto
    cfg1 = f"{BASE}/inference_config.json."
    cfg2 = f"{BASE}/inference_config.json"
    config_picked = cfg1 if os.path.isfile(cfg1) else (cfg2 if os.path.isfile(cfg2) else "")

    for repo in _find_musetalk_repo_candidates():
        inf = os.path.join(repo, "scripts", "inference.py")
        if os.path.isfile(inf):
            return repo, config_picked

    raise RuntimeError("No encuentro MuseTalk repo con scripts/inference.py en el volumen.")

def _ensure_repo_config(repo_root: str, config_src: str) -> str:
    """
    Asegura que dentro del repo exista inference_config.json
    SIN duplicar carpetas pesadas: solo copia ese JSON.
    """
    if not config_src or not os.path.isfile(config_src):
        # Si no hay config en el volumen, intentamos usar el del repo
        candidate = os.path.join(repo_root, "inference_config.json")
        if os.path.isfile(candidate):
            return candidate
        raise RuntimeError("No encuentro inference_config.json (ni en volumen ni en repo).")

    dst = os.path.join(repo_root, "inference_config.json")
    if os.path.isfile(dst):
        return dst

    # copiar un archivo pequeño (no llena disco)
    _run(["bash", "-lc", f"cp -f '{config_src}' '{dst}'"], env=_clean_env())
    return dst

# ---------- XTTS ----------
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
    # por si acaso pregunta, respondemos y además está COQUI_TOS_AGREED=1
    _run(cmd, env=_clean_env(), stdin_text="y\n")

# ---------- MuseTalk inference ----------
def _musetalk_infer(repo_root: str, config_json: str, input_mp4: str, audio_wav: str) -> str:
    _require_dir(repo_root, "MuseTalk repo_root")

    # inputs dentro del repo (MuseTalk los espera ahí)
    inputs_dir = os.path.join(repo_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    # CLAVE: PYTHONPATH al repo para que "import musetalk" funcione
    env = _clean_env({"PYTHONPATH": repo_root})

    cmd = [
        SYS_PY, "-u", "scripts/inference.py",
        "--inference_config", os.path.basename(config_json),
        "--bbox_shift", "0",
        "--use_float16"
    ]
    _run(cmd, cwd=repo_root, env=env)

    # Busca resultados (varía según fork; probamos varios)
    candidates = [
        os.path.join(repo_root, "results", "v15"),
        os.path.join(repo_root, "results"),
        os.path.join(repo_root, "output"),
        os.path.join(repo_root, "outputs"),
    ]
    mp4s = []
    for d in candidates:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".mp4"):
                    mp4s.append(os.path.join(d, f))

    if not mp4s:
        raise RuntimeError("MuseTalk no produjo mp4 (no encontré outputs).")

    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]

# ---------- Main mode ----------
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

    repo_root, config_src = _pick_musetalk_repo()
    config_json = _ensure_repo_config(repo_root, config_src)

    with tempfile.TemporaryDirectory() as td:
        in_mp4  = os.path.join(td, "in.mp4")
        tts_wav = os.path.join(td, "tts.wav")

        if video_url:
            _download_to_file(video_url, in_mp4)
        else:
            _b64_to_file(str(video_b64), in_mp4)

        _tts_make_wav(text=text, voice=voice, lang=lang, out_wav=tts_wav)
        out_mp4_path = _musetalk_infer(repo_root=repo_root, config_json=config_json, input_mp4=in_mp4, audio_wav=tts_wav)

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
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "MUSE_REPO_ROOT": repo_root,
            "MUSE_CONFIG_JSON": config_json,
        }
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode in ("echo", "debug"):
            try:
                repo_root, config_src = _pick_musetalk_repo()
                cfg = _ensure_repo_config(repo_root, config_src)
                muse_ok = True
            except Exception:
                repo_root, cfg = "", ""
                muse_ok = False

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
                    "musetalk_repo_ok": muse_ok,
                    "musetalk_scripts_inference_exists": os.path.isfile(os.path.join(repo_root, "scripts", "inference.py")) if repo_root else False,
                    "musetalk_config_exists": bool(cfg) and os.path.isfile(cfg),
                },
                "paths": {
                    "VOICES_DIR": VOICES_DIR,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                    "MUSE_REPO_ROOT": repo_root,
                    "MUSE_CONFIG_JSON": cfg,
                }
            }

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})
