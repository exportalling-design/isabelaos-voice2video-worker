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
# ENV hardening (no contaminar)
# ----------------------------
os.environ.pop("PYTHONPATH", None)
os.environ.pop("PYTHONHOME", None)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ Auto-acepta CPML (si ya lo pusiste en Dockerfile, perfecto)
os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("TTS_USE_GPU", "1")

SYS_PY = "/usr/local/bin/python3"


# ----------------------------
# Base detect
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

# ----------------------------
# Paths (NO inventar)
# ----------------------------
VOICES_DIR = os.environ.get("VOICES_DIR") or f"{BASE}/voices"
FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV") or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV   = os.environ.get("MALE_REF_WAV")   or f"{VOICES_DIR}/male_ref.wav"

# ✅ MuseTalk en tu volumen normalmente es /runpod-volume/musetalk_ok
def _detect_muse_root(base: str) -> str:
    # prioridad: musetalk_ok
    cands = [
        os.environ.get("MUSE_ROOT", "").strip(),
        f"{base}/musetalk_ok",
        f"{base}/MuseTalk",
        f"{base}/MuseTalk_ok",
        f"{base}/MuseTalkOK",
    ]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    # si no existe, devuelve el primero razonable (para echo)
    return (os.environ.get("MUSE_ROOT") or f"{base}/musetalk_ok").strip()

MUSE_ROOT = _detect_muse_root(BASE)


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
    env.setdefault("TTS_USE_GPU", "1")
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

def _safe_listdir(path: str, limit: int = 200):
    try:
        items = sorted(os.listdir(path))
        return items[:limit]
    except Exception as e:
        return [f"[LIST_ERROR] {e}"]

def _scan_musetalk(root: str, max_hits: int = 200):
    out = {
        "root": root,
        "root_exists": os.path.isdir(root),
        "top_level": _safe_listdir(root, limit=200),
        "found_configs": [],
        "found_py_candidates": [],
    }
    if not os.path.isdir(root):
        return out

    # busca inference_config.json* cerca
    cfg_hits = []
    py_hits = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            rel_depth = os.path.relpath(dirpath, root).count(os.sep)
            if rel_depth > 4:
                continue

            for fn in filenames:
                low = fn.lower()
                full = os.path.join(dirpath, fn)

                if low.startswith("inference_config") and low.endswith(".json"):
                    cfg_hits.append(full)

                if low.endswith(".py"):
                    if ("inference" in low) or low.startswith("infer") or low in ("app.py", "run.py", "demo.py"):
                        py_hits.append(full)

            if len(cfg_hits) >= max_hits and len(py_hits) >= max_hits:
                break
    except Exception as e:
        out["walk_error"] = str(e)

    out["found_configs"] = cfg_hits[:max_hits]
    out["found_py_candidates"] = py_hits[:max_hits]
    return out

def _find_file_by_names(root: str, names: List[str]) -> Optional[str]:
    for n in names:
        p = os.path.join(root, n)
        if os.path.isfile(p):
            return p
    return None

def _find_musetalk_entry(root: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Devuelve (entry_py_abs, config_json_abs).
    NO asume scripts/inference.py; lo busca.
    """
    if not os.path.isdir(root):
        return None, None

    # 1) entry candidates (orden de prioridad)
    entry = _find_file_by_names(root, [
        "scripts/inference.py",
        "inference.py",
        "scripts/infer.py",
        "infer.py",
        "app.py",
        "run.py",
        "demo.py",
    ])

    # 2) si no aparece, hacemos scan corto y agarramos el mejor candidato
    cfg = _find_file_by_names(root, ["inference_config.json"])
    if not entry or not cfg:
        scan = _scan_musetalk(root, max_hits=200)
        if not cfg and scan.get("found_configs"):
            # el primero encontrado
            cfg = scan["found_configs"][0]
        if not entry and scan.get("found_py_candidates"):
            # preferir el que incluya "inference.py"
            cand = scan["found_py_candidates"]
            best = None
            for p in cand:
                if os.path.basename(p).lower() == "inference.py":
                    best = p
                    break
            entry = best or cand[0]

    # 3) fallback: si no hay config en root, busca uno en BASE (por si lo dejaste fuera)
    if not cfg:
        # hay veces que el archivo quedó como "inference_config.json."
        for nm in ["inference_config.json", "inference_config.json."]:
            p = os.path.join(BASE, nm)
            if os.path.isfile(p):
                cfg = p
                break

    return entry, cfg


# ----------------------------
# XTTS via Coqui TTS (realista)
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

    # ✅ por si intenta preguntar y/n (aunque COQUI_TOS_AGREED=1 debería bastar)
    _run(cmd, env=_clean_env(), stdin_text="y\n")


# ----------------------------
# MuseTalk inference
# ----------------------------
def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    _require_dir(MUSE_ROOT, "MUSE_ROOT (MuseTalk folder)")

    entry_py, cfg_json = _find_musetalk_entry(MUSE_ROOT)
    if not entry_py:
        raise RuntimeError(f"No encontré entrypoint .py de MuseTalk dentro de: {MUSE_ROOT}. Usa mode=scan_musetalk.")
    if not cfg_json:
        raise RuntimeError(f"No encontré inference_config.json (ni en {MUSE_ROOT} ni en {BASE}). Usa mode=scan_musetalk.")

    # inputs/
    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    # ✅ ejecuta el entry real
    cmd = [
        SYS_PY, "-u", entry_py,
        "--inference_config", cfg_json,
        "--bbox_shift", "0",
        "--use_float16"
    ]
    _run(cmd, cwd=MUSE_ROOT, env=_clean_env())

    # resultados típicos
    results_dir = os.path.join(MUSE_ROOT, "results", "v15")
    if not os.path.isdir(results_dir):
        # fallback: busca cualquier results/*
        results_parent = os.path.join(MUSE_ROOT, "results")
        if os.path.isdir(results_parent):
            # agarra el subfolder más nuevo
            subs = [os.path.join(results_parent, d) for d in os.listdir(results_parent)]
            subs = [d for d in subs if os.path.isdir(d)]
            subs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            results_dir = subs[0] if subs else results_dir

    if not os.path.isdir(results_dir):
        raise RuntimeError(f"No results dir: {results_dir}")

    mp4s = []
    for fn in os.listdir(results_dir):
        if fn.lower().endswith(".mp4"):
            mp4s.append(os.path.join(results_dir, fn))

    if not mp4s:
        raise RuntimeError(f"MuseTalk no produjo mp4 en: {results_dir}")

    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]


# ----------------------------
# Main job
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
        },
        "env": {
            "RUNPOD_VOLUME_PATH": os.environ.get("RUNPOD_VOLUME_PATH"),
            "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
            "TTS_USE_GPU": os.environ.get("TTS_USE_GPU"),
        }
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode in ("echo", "debug"):
            entry_py, cfg_json = _find_musetalk_entry(MUSE_ROOT)
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
                    "muse_root_exists": os.path.isdir(MUSE_ROOT),
                    "voices_dir_exists": os.path.isdir(VOICES_DIR),
                    "female_ref_exists": _exists_file(FEMALE_REF_WAV),
                    "male_ref_exists": _exists_file(MALE_REF_WAV),
                    "musetalk_entry_found": bool(entry_py),
                    "musetalk_config_found": bool(cfg_json),
                },
                "paths": {
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                    "MUSE_ENTRY_PY": entry_py,
                    "MUSE_CONFIG_JSON": cfg_json,
                }
            }

        if mode in ("scan_musetalk", "scan"):
            return {
                "ok": True,
                "msg": "SCAN_OK",
                "base": BASE,
                "MUSE_ROOT": MUSE_ROOT,
                "scan": _scan_musetalk(MUSE_ROOT, max_hits=200),
            }

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|scan_musetalk|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()


runpod.serverless.start({"handler": handler})
