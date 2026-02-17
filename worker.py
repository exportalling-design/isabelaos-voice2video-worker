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

# ✅ Auto-acepta CPML
os.environ.setdefault("COQUI_TOS_AGREED", "1")

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
# Paths
# ----------------------------
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

def _find_any(root: str, names: List[str]) -> Optional[str]:
    for n in names:
        p = os.path.join(root, n)
        if os.path.isfile(p):
            return p
    return None

def _scan_tree_for(root: str, want_ext: str = ".py", max_hits: int = 300, max_depth: int = 6) -> List[str]:
    hits = []
    if not os.path.isdir(root):
        return hits
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.lower().endswith(want_ext):
                hits.append(os.path.join(dirpath, fn))
                if len(hits) >= max_hits:
                    return hits
    return hits

def _looks_like_repo_root(p: str) -> bool:
    if not os.path.isdir(p):
        return False
    # marcadores típicos de repo MuseTalk
    if os.path.isdir(os.path.join(p, "scripts")):
        return True
    if os.path.isfile(os.path.join(p, "inference_config.json")) or os.path.isfile(os.path.join(p, "inference_config.json.")):
        return True
    if os.path.isdir(os.path.join(p, "checkpoints")) or os.path.isdir(os.path.join(p, "models")):
        return True
    return False

def _detect_muse_root(base: str) -> str:
    # Si el user lo setea, lo respetamos, pero igual validamos después
    env_root = (os.environ.get("MUSE_ROOT") or "").strip()
    cands = []
    if env_root:
        cands.append(env_root)

    # tus folders conocidos
    cands += [
        f"{base}/musetalk_ok",
        f"{base}/musetalk_ok_persist",
        f"{base}/MuseTalk",
        f"{base}/MuseTalk_ok",
        f"{base}/projects/MuseTalk",
    ]

    # elegí el primero que "parezca repo"
    for c in cands:
        if c and _looks_like_repo_root(c):
            return c

    # fallback: el primero existente
    for c in cands:
        if c and os.path.isdir(c):
            return c

    return env_root or f"{base}/musetalk_ok"

MUSE_ROOT = _detect_muse_root(BASE)

def _find_musetalk_entry_and_cfg(root: str) -> Tuple[Optional[str], Optional[str]]:
    """
    NO usa site-packages.
    Busca entry real dentro del repo: scripts/inference.py, inference.py, etc.
    """
    if not os.path.isdir(root):
        return None, None

    # config: preferí dentro de root
    cfg = _find_any(root, ["inference_config.json", "inference_config.json."])
    if not cfg:
        # fallback: en BASE (como tu caso)
        cfg = _find_any(BASE, ["inference_config.json", "inference_config.json."])

    # entry candidates de alta prioridad
    priority = [
        os.path.join(root, "scripts", "inference.py"),
        os.path.join(root, "inference.py"),
        os.path.join(root, "scripts", "infer.py"),
        os.path.join(root, "infer.py"),
        os.path.join(root, "app.py"),
        os.path.join(root, "run.py"),
        os.path.join(root, "demo.py"),
    ]
    for p in priority:
        if os.path.isfile(p):
            return p, cfg

    # scan y filtrar: NO site-packages
    pys = _scan_tree_for(root, want_ext=".py", max_hits=500, max_depth=7)
    pys = [p for p in pys if "site-packages" not in p.replace("\\", "/")]
    if not pys:
        return None, cfg

    # preferir nombres con inference
    def score(p: str) -> int:
        b = os.path.basename(p).lower()
        s = 0
        if "inference" in b: s += 50
        if b in ("inference.py",): s += 100
        if "/scripts/" in p.replace("\\", "/"): s += 30
        return s

    pys.sort(key=lambda p: (score(p), os.path.getmtime(p)), reverse=True)
    return pys[0], cfg


# ----------------------------
# XTTS via Coqui TTS (CPU to avoid ECC)
# ----------------------------
def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")

    # ✅ ECC fix: forzar TTS en CPU
    # (MuseTalk seguirá usando GPU normalmente)
    env = _clean_env({
        "TTS_USE_GPU": "0",          # <<<<<< CLAVE
        "COQUI_TOS_AGREED": "1",
        # cache en volumen para no re-descargar tanto (si ya existe, lo reutiliza)
        "XDG_DATA_HOME": f"{BASE}/.local",
    })

    cmd = [
        SYS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav
    ]

    _run(cmd, env=env, stdin_text="y\n")


# ----------------------------
# MuseTalk inference
# ----------------------------
def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    _require_dir(MUSE_ROOT, "MUSE_ROOT (MuseTalk repo folder)")

    entry_py, cfg_json = _find_musetalk_entry_and_cfg(MUSE_ROOT)
    if not entry_py:
        raise RuntimeError(f"No encontré entrypoint real de MuseTalk dentro de: {MUSE_ROOT}. Usá mode=scan_musetalk.")
    if not cfg_json:
        raise RuntimeError(f"No encontré inference_config.json(.). Está ni en {MUSE_ROOT} ni en {BASE}.")

    inputs_dir = os.path.join(MUSE_ROOT, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"], env=_clean_env())
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"], env=_clean_env())

    cmd = [
        SYS_PY, "-u", entry_py,
        "--inference_config", cfg_json,
        "--bbox_shift", "0",
        "--use_float16"
    ]
    _run(cmd, cwd=MUSE_ROOT, env=_clean_env())

    # results típicos
    results_dir = os.path.join(MUSE_ROOT, "results", "v15")
    if not os.path.isdir(results_dir):
        results_parent = os.path.join(MUSE_ROOT, "results")
        if os.path.isdir(results_parent):
            subs = [os.path.join(results_parent, d) for d in os.listdir(results_parent)]
            subs = [d for d in subs if os.path.isdir(d)]
            subs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            results_dir = subs[0] if subs else results_dir

    if not os.path.isdir(results_dir):
        raise RuntimeError(f"No results dir: {results_dir}")

    mp4s = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.lower().endswith(".mp4")]
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
            "TTS_USE_GPU_effective": "0 (forced in worker)",
        }
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or "").strip().lower()

        if mode in ("echo", "debug"):
            entry_py, cfg_json = _find_musetalk_entry_and_cfg(MUSE_ROOT)
            return {
                "ok": True,
                "msg": "ECHO_OK",
                "python": SYS_PY,
                "base": BASE,
                "env": {
                    "RUNPOD_VOLUME_PATH": os.environ.get("RUNPOD_VOLUME_PATH"),
                    "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED"),
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
            scan = {
                "root": MUSE_ROOT,
                "root_exists": os.path.isdir(MUSE_ROOT),
                "top_level": _safe_listdir(MUSE_ROOT, 200),
                "py_hits_sample": [],
                "cfg_hits": [],
            }
            if os.path.isdir(MUSE_ROOT):
                pys = _scan_tree_for(MUSE_ROOT, ".py", max_hits=300, max_depth=7)
                pys = [p for p in pys if "site-packages" not in p.replace("\\", "/")]
                scan["py_hits_sample"] = pys[:80]
                cfgs = []
                for name in ("inference_config.json", "inference_config.json."):
                    p = os.path.join(MUSE_ROOT, name)
                    if os.path.isfile(p):
                        cfgs.append(p)
                for name in ("inference_config.json", "inference_config.json."):
                    p = os.path.join(BASE, name)
                    if os.path.isfile(p) and p not in cfgs:
                        cfgs.append(p)
                scan["cfg_hits"] = cfgs

            return {"ok": True, "msg": "SCAN_OK", "base": BASE, "scan": scan}

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|scan_musetalk|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()


runpod.serverless.start({"handler": handler})
