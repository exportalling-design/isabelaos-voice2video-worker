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

# --- Hardening base ---
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("COQUI_TOS_AGREED", "1")  # auto-acepta TOS en serverless

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
# Utils
# ----------------------------
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
        tail = (p.stdout or "")[-14000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout or ""

# ----------------------------
# MuseTalk autodetect (repo real)
# ----------------------------
def _is_musetalk_repo(dir_path: str) -> bool:
    # Repo real debe tener scripts/inference.py
    return _exists_file(os.path.join(dir_path, "scripts", "inference.py"))

def _find_musetalk_candidates(base: str, max_depth: int = 4) -> List[str]:
    """
    Busca folders que parezcan repo MuseTalk dentro del volumen.
    Limitado por depth para no caminar infinito.
    """
    hits = []
    base = base.rstrip("/")
    if not _exists_dir(base):
        return hits

    # candidatos directos (rápidos)
    direct = [
        os.environ.get("MUSE_ROOT") or "",
        f"{base}/MuseTalk",
        f"{base}/musetalk",
        f"{base}/musetalk_ok",      # OJO: normalmente es venv, pero igual lo revisamos
        f"{base}/musetalk_ok_persist",
        f"{base}/projects/MuseTalk",
        f"{base}/projects/musetalk",
    ]
    for d in direct:
        if d and _exists_dir(d) and _is_musetalk_repo(d):
            hits.append(d)

    # walk limitado
    def depth(path: str) -> int:
        rel = os.path.relpath(path, base)
        if rel == ".":
            return 0
        return rel.count(os.sep) + 1

    for root, dirs, files in os.walk(base):
        if depth(root) > max_depth:
            dirs[:] = []
            continue
        if "scripts" in dirs:
            if _is_musetalk_repo(root):
                hits.append(root)
                # no hace falta seguir bajando dentro de este repo
                dirs[:] = []
                continue

    # unique
    uniq = []
    seen = set()
    for h in hits:
        if h not in seen:
            uniq.append(h)
            seen.add(h)
    return uniq

def _pick_musetalk_root(base: str) -> Tuple[str, List[str]]:
    cands = _find_musetalk_candidates(base)
    if not cands:
        # Si no hay repo, devolvemos "" y lista vacía
        return "", []
    # preferir el que NO sea venv típico (bin/lib/include)
    # pero si solo hay uno, usarlo.
    def score(p: str) -> int:
        s = 0
        if _exists_file(os.path.join(p, "scripts", "inference.py")):
            s += 100
        if _exists_file(os.path.join(p, "inference_config.json")) or _exists_file(os.path.join(p, "inference_config.json.")):
            s += 10
        # penaliza si parece venv
        if _exists_file(os.path.join(p, "pyvenv.cfg")) and _exists_dir(os.path.join(p, "bin")):
            s -= 5
        return s
    cands.sort(key=score, reverse=True)
    return cands[0], cands

def _pick_musetalk_config(base: str, muse_root: str) -> str:
    # Tu caso: /runpod-volume/inference_config.json. (con punto)
    options = [
        os.environ.get("MUSE_CONFIG_JSON") or "",
        f"{base}/inference_config.json",
        f"{base}/inference_config.json.",
        os.path.join(muse_root, "inference_config.json") if muse_root else "",
        os.path.join(muse_root, "inference_config.json.") if muse_root else "",
    ]
    for p in options:
        if p and _exists_file(p):
            return p
    # fallback: vacío
    return ""

# ----------------------------
# XTTS: generar WAV (con fallback CPU si ECC)
# ----------------------------
def _tts_make_wav_xtts(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")

    cmd = [
        SYS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav
    ]

    # intento 1: como venga (GPU si TTS_USE_GPU=1)
    try:
        _run(cmd, env=_clean_env(), stdin_text="y\n")
        return
    except Exception as e:
        msg = str(e)
        # Si es ECC o error CUDA, reintentar CPU
        if ("uncorrectable ECC" in msg) or ("CUDA error" in msg) or ("cuda" in msg.lower()):
            env_cpu = _clean_env({
                "TTS_USE_GPU": "0",
                "CUDA_VISIBLE_DEVICES": ""  # fuerza CPU
            })
            _run(cmd, env=env_cpu, stdin_text="y\n")
            return
        raise

# ----------------------------
# MuseTalk infer
# ----------------------------
def _musetalk_infer(muse_root: str, config_json: str, input_mp4: str, audio_wav: str) -> str:
    _require_dir(muse_root, "MUSE_ROOT (MuseTalk repo folder)")
    runner = os.path.join(muse_root, "scripts", "inference.py")
    _require_file(runner, "MuseTalk scripts/inference.py")

    if not config_json:
        raise RuntimeError(
            "No se encontró inference_config.json(.). "
            f"Busqué en {BASE} y en {muse_root}. "
            "Asegurate que exista /runpod-volume/inference_config.json. (o sin punto)."
        )

    inputs_dir = os.path.join(muse_root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    cmd = [
        SYS_PY, "-u", runner,
        "--inference_config", config_json,
        "--bbox_shift", "0",
        "--use_float16"
    ]
    _run(cmd, cwd=muse_root, env=_clean_env())

    # resultados: algunos setups usan results/v15, otros results/...
    results_base = os.path.join(muse_root, "results")
    if not _exists_dir(results_base):
        raise RuntimeError(f"MuseTalk no creó carpeta results: {results_base}")

    # busca mp4 más nuevo en results/**
    newest = ("", 0.0)
    for root, _, files in os.walk(results_base):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                p = os.path.join(root, fn)
                mt = os.path.getmtime(p)
                if mt > newest[1]:
                    newest = (p, mt)

    if not newest[0]:
        raise RuntimeError("MuseTalk no produjo ningún mp4 en results/")

    return newest[0]

# ----------------------------
# Main mode: voice_to_video
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

    muse_root, muse_cands = _pick_musetalk_root(BASE)
    if not muse_root:
        raise RuntimeError(
            "No encontré el repo de MuseTalk en el volumen. "
            "Necesito un folder que tenga scripts/inference.py. "
            f"Candidatos revisados: {muse_cands}"
        )

    config_json = _pick_musetalk_config(BASE, muse_root)

    with tempfile.TemporaryDirectory() as td:
        in_mp4  = os.path.join(td, "in.mp4")
        tts_wav = os.path.join(td, "tts.wav")

        if video_url:
            _download_to_file(video_url, in_mp4)
        else:
            _b64_to_file(str(video_b64), in_mp4)

        _tts_make_wav_xtts(text=text, voice=voice, lang=lang, out_wav=tts_wav)
        out_mp4_path = _musetalk_infer(muse_root=muse_root, config_json=config_json, input_mp4=in_mp4, audio_wav=tts_wav)

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
            "MUSE_ROOT": muse_root,
            "MUSE_CONFIG_JSON": config_json,
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
        }
    }

# ----------------------------
# Debug modes
# ----------------------------
def scan_musetalk() -> Dict[str, Any]:
    muse_root, muse_cands = _pick_musetalk_root(BASE)
    cfg = _pick_musetalk_config(BASE, muse_root) if muse_root else _pick_musetalk_config(BASE, "")
    sample_py = []

    # muestra pocos .py dentro del root (si existe)
    if muse_root:
        for p in [
            os.path.join(muse_root, "scripts", "inference.py"),
            os.path.join(muse_root, "inference_config.json"),
            os.path.join(muse_root, "inference_config.json."),
        ]:
            if _exists_file(p):
                sample_py.append(p)

    return {
        "ok": True,
        "msg": "SCAN_OK",
        "base": BASE,
        "scan": {
            "muse_root_picked": muse_root,
            "muse_candidates": muse_cands[:30],
            "config_picked": cfg,
            "important_hits": sample_py,
        }
    }

def echo() -> Dict[str, Any]:
    muse_root, muse_cands = _pick_musetalk_root(BASE)
    cfg = _pick_musetalk_config(BASE, muse_root) if muse_root else _pick_musetalk_config(BASE, "")

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
            "muse_root_picked_exists": _exists_dir(muse_root) if muse_root else False,
            "muse_runner_exists": _exists_file(os.path.join(muse_root, "scripts", "inference.py")) if muse_root else False,
            "config_exists": _exists_file(cfg) if cfg else False,
        },
        "paths": {
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "MUSE_ROOT_PICKED": muse_root,
            "MUSE_CONFIG_JSON": cfg,
        },
        "muse_candidates_sample": muse_cands[:15],
    }

# ----------------------------
# Handler
# ----------------------------
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
