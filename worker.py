# /app/worker.py
import os
import gc
import time
import base64
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, List, Tuple

import runpod

# ----------------------------
# Hardening env
# ----------------------------
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("COQUI_TOS_AGREED", "1")

SYS_PY = "/usr/local/bin/python3"


# ----------------------------
# Path helpers
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

VOICES_DIR = os.environ.get("VOICES_DIR") or f"{BASE}/voices"
FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV = os.environ.get("MALE_REF_WAV") or f"{VOICES_DIR}/male_ref.wav"


def _exists_file(p: str) -> bool:
    return bool(p) and os.path.isfile(p)

def _exists_dir(p: str) -> bool:
    return bool(p) and os.path.isdir(p)

def _require_file(path: str, label: str):
    if not _exists_file(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _require_dir(path: str, label: str):
    if not _exists_dir(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass


# ----------------------------
# Base64 utils
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
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())


# ----------------------------
# Subprocess runner
# ----------------------------
def _clean_env(extra: Dict[str, str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("COQUI_TOS_AGREED", "1")
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
        tail = out[-14000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return out


# ----------------------------
# MuseTalk detection (repo real)
# ----------------------------
def _find_candidates() -> List[str]:
    # lugares típicos que ya vimos en tus logs
    cands = [
        os.environ.get("MUSE_ROOT", "").strip(),
        f"{BASE}/musetalk_ok",                 # tu venv (no repo)
        f"{BASE}/MuseTalk",                   # repo
        f"{BASE}/volume_old/MuseTalk",         # repo real que apareció en scan
        f"{BASE}/volume_old/MuseTalk_repo_tmp",
    ]
    out = []
    for p in cands:
        if p and p not in out:
            out.append(p)
    return out

def _is_musetalk_repo(path: str) -> bool:
    if not _exists_dir(path):
        return False
    # validamos que exista el script y el paquete musetalk/
    return _exists_file(os.path.join(path, "scripts", "inference.py")) and _exists_dir(os.path.join(path, "musetalk"))

def _pick_musetalk_repo() -> Optional[str]:
    for p in _find_candidates():
        if _is_musetalk_repo(p):
            return p
    # fallback: buscar recursivo liviano (solo 2 niveles) en /runpod-volume
    root = BASE if _exists_dir(BASE) else "/runpod-volume"
    try:
        for a in os.listdir(root):
            pa = os.path.join(root, a)
            if not os.path.isdir(pa):
                continue
            # 1 nivel adentro
            if _is_musetalk_repo(pa):
                return pa
            try:
                for b in os.listdir(pa):
                    pb = os.path.join(pa, b)
                    if _is_musetalk_repo(pb):
                        return pb
            except Exception:
                pass
    except Exception:
        pass
    return None

def _pick_musetalk_config(repo_root: str) -> Optional[str]:
    # Preferimos config dentro del repo si existe
    p1 = os.path.join(repo_root, "inference_config.json")
    if _exists_file(p1):
        return p1

    # A vos te aparece: /runpod-volume/inference_config.json. (con punto)
    # y también /runpod-volume/volume_old/MuseTalk/inference_config.json
    candidates = [
        os.path.join(BASE, "inference_config.json"),
        os.path.join(BASE, "inference_config.json."),
        os.path.join(repo_root, "inference_config.json."),
        os.path.join(repo_root, "inference_config.json"),
    ]
    for c in candidates:
        if _exists_file(c):
            return c

    # último intento: buscar cualquier inference_config*.json*
    try:
        hits = []
        for dirpath, dirnames, filenames in os.walk(BASE):
            for fn in filenames:
                low = fn.lower()
                if "inference_config" in low and low.startswith("inference_config") and "json" in low:
                    hits.append(os.path.join(dirpath, fn))
            # no hagas walk infinito
            if len(hits) >= 5:
                break
        if hits:
            hits.sort()
            return hits[0]
    except Exception:
        pass
    return None

def _ensure_repo_symlinks(repo_root: str, config_picked: str):
    """
    MuseTalk script normalmente espera:
      --inference_config inference_config.json   (en cwd)
    Si tu config real es inference_config.json. con punto,
    creamos un symlink/copia a inference_config.json dentro del repo.
    """
    target = os.path.join(repo_root, "inference_config.json")
    if os.path.abspath(config_picked) == os.path.abspath(target):
        return

    # Intento symlink -> si no, copy
    try:
        _run(["bash", "-lc", f"ln -sf '{config_picked}' '{target}'"])
    except Exception:
        try:
            _run(["bash", "-lc", f"cp -f '{config_picked}' '{target}'"])
        except Exception as e:
            raise RuntimeError(f"No pude preparar inference_config.json en repo. picked={config_picked} err={e}")

def _musetalk_infer(repo_root: str, input_mp4: str, audio_wav: str) -> str:
    _require_dir(repo_root, "MuseTalk repo_root")
    _require_file(os.path.join(repo_root, "scripts", "inference.py"), "MuseTalk scripts/inference.py")
    _require_dir(os.path.join(repo_root, "musetalk"), "MuseTalk package folder musetalk/")

    config_picked = _pick_musetalk_config(repo_root)
    if not config_picked:
        raise RuntimeError("No encontré inference_config.json para MuseTalk")
    _ensure_repo_symlinks(repo_root, config_picked)

    # MuseTalk usa carpeta inputs
    inputs_dir = os.path.join(repo_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"])
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"])

    # ✅ CLAVE: PYTHONPATH debe incluir repo_root para que "import musetalk" funcione
    env = _clean_env({
        "PYTHONPATH": repo_root,
    })

    cmd = [
        SYS_PY, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
        "--bbox_shift", "0",
        "--use_float16"
    ]
    _run(cmd, cwd=repo_root, env=env)

    # Resultados típicos (en repo: results/v15)
    results_dir = os.path.join(repo_root, "results", "v15")
    if not os.path.isdir(results_dir):
        # fallback: buscar mp4 en results
        res_root = os.path.join(repo_root, "results")
        if os.path.isdir(res_root):
            mp4s = []
            for dp, dn, fn in os.walk(res_root):
                for f in fn:
                    if f.endswith(".mp4"):
                        mp4s.append(os.path.join(dp, f))
            if mp4s:
                mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return mp4s[0]
        raise RuntimeError(f"No results dir: {results_dir}")

    mp4s = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".mp4")]
    if not mp4s:
        raise RuntimeError("MuseTalk no produjo mp4 (results/v15 vacío)")

    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]


# ----------------------------
# XTTS (Coqui TTS)
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

    # Coqui a veces intenta preguntar el ToS -> respondemos "y"
    _run(cmd, env=_clean_env(), stdin_text="y\n")


# ----------------------------
# Main pipeline: voice_to_video
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

    repo_root = _pick_musetalk_repo()
    if not repo_root:
        raise RuntimeError("No encontré el repo de MuseTalk (debe tener scripts/inference.py y carpeta musetalk/)")

    with tempfile.TemporaryDirectory() as td:
        in_mp4 = os.path.join(td, "in.mp4")
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
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "MUSE_REPO_ROOT": repo_root,
        }
    }


# ----------------------------
# Debug modes
# ----------------------------
def _scan() -> Dict[str, Any]:
    repo = _pick_musetalk_repo()
    cfg = _pick_musetalk_config(repo) if repo else None
    imp_hits = []
    if repo:
        # cosas importantes que deberían existir
        for p in [
            os.path.join(repo, "scripts", "inference.py"),
            os.path.join(repo, "musetalk"),
            os.path.join(repo, "inference_config.json"),
            os.path.join(repo, "inference_config.json."),
        ]:
            if os.path.exists(p):
                imp_hits.append(p)

    return {
        "ok": True,
        "msg": "SCAN_OK",
        "base": BASE,
        "scan": {
            "repo_picked": repo,
            "config_picked": cfg,
            "important_hits": imp_hits,
            "top_level_repo": os.listdir(repo)[:40] if repo and os.path.isdir(repo) else None,
        }
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode in ("echo", "debug"):
            repo = _pick_musetalk_repo()
            cfg = _pick_musetalk_config(repo) if repo else None
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
                    "voices_dir_exists": _exists_dir(VOICES_DIR),
                    "female_ref_exists": _exists_file(FEMALE_REF_WAV),
                    "male_ref_exists": _exists_file(MALE_REF_WAV),
                    "muse_repo_exists": bool(repo and os.path.isdir(repo)),
                    "muse_scripts_inference_exists": bool(repo and os.path.isfile(os.path.join(repo, "scripts", "inference.py"))),
                    "muse_config_exists": bool(cfg and os.path.isfile(cfg)),
                },
                "paths": {
                    "VOICES_DIR": VOICES_DIR,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                    "MUSE_REPO_PICKED": repo,
                    "MUSE_CONFIG_PICKED": cfg,
                },
            }

        if mode in ("scan",):
            return _scan()

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|scan|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()

runpod.serverless.start({"handler": handler})
