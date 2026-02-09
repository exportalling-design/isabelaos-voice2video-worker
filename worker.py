# /app/worker.py
import os
import gc
import time
import base64
import binascii
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, List

import runpod

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _hard_cleanup():
    try:
        gc.collect()
    except Exception:
        pass


def _norm_path(p: Optional[str]) -> str:
    p = (p or "").strip()
    if not p:
        return p
    if p.startswith("runpod-volume/"):
        p = "/" + p
    while "//" in p:
        p = p.replace("//", "/")
    return p


def _exists_dir(p: str) -> bool:
    p = _norm_path(p)
    return bool(p) and os.path.isdir(p)


def _run(cmd: list, cwd: str = None, timeout: int = 1200):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if p.returncode != 0:
        tail = (p.stdout or "")[-12000:]
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{tail}")
    return p.stdout or ""


def _safe(cmd: list, cwd: str = None, timeout: int = 30) -> str:
    try:
        return _run(cmd, cwd=cwd, timeout=timeout)
    except Exception as e:
        return f"[FAILED] {e}"


def _detect_base() -> str:
    candidates = []
    rp = _norm_path(os.environ.get("RUNPOD_VOLUME_PATH"))
    if rp:
        candidates.append(rp)

    for k in ("VOLUME_PATH", "BASE"):
        v = _norm_path(os.environ.get(k))
        if v:
            candidates.append(v)

    candidates += ["/runpod-volume", "/workspace", "/mnt", "/data", "/volume", "/workspace/runpod-volume"]

    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mp = _norm_path(parts[1])
                    if mp and mp not in candidates:
                        candidates.append(mp)
    except Exception:
        pass

    for base in [c for c in candidates if c]:
        if _exists_dir(os.path.join(base, "MuseTalk")) and _exists_dir(os.path.join(base, "voices")):
            return base

    return rp or "/runpod-volume"


BASE = _detect_base()

MUSE_ROOT  = _norm_path(os.environ.get("MUSE_ROOT"))  or f"{BASE}/MuseTalk"
VOICES_DIR = _norm_path(os.environ.get("VOICES_DIR")) or f"{BASE}/voices"

FEMALE_REF_WAV = _norm_path(os.environ.get("FEMALE_REF_WAV")) or f"{VOICES_DIR}/female_ref.wav"
MALE_REF_WAV   = _norm_path(os.environ.get("MALE_REF_WAV"))   or f"{VOICES_DIR}/male_ref.wav"

# Optional (no lo usamos como python)
TTS_BIN = _norm_path(os.environ.get("TTS_BIN")) or f"{BASE}/xtts_env/bin/tts"


def _pick_runnable_python(candidates: List[str]) -> Dict[str, Any]:
    """
    Returns: {"ok": bool, "path": str, "tested": [...], "why": "..."}
    """
    tested = []
    for p in candidates:
        p = _norm_path(p)
        if not p:
            continue
        # intenta correr python -V (esto falla si es symlink roto)
        out = _safe([p, "-V"], timeout=15)
        tested.append({"path": p, "out": out[:300]})
        if not out.startswith("[FAILED]"):
            return {"ok": True, "path": p, "tested": tested, "why": "python -V ok"}
    return {"ok": False, "path": candidates[0] if candidates else "", "tested": tested, "why": "no runnable python found"}


# OJO: aquí dejamos de confiar en "python" a ciegas
TTS_PICK = _pick_runnable_python([
    _norm_path(os.environ.get("TTS_PY")) or f"{BASE}/xtts_env/bin/python",
    f"{BASE}/xtts_env/bin/python",
    f"{BASE}/xtts_env/bin/python3",
    f"{BASE}/xtts_env/bin/python3.11",
])

MUSE_PICK = _pick_runnable_python([
    _norm_path(os.environ.get("MUSE_PY")) or f"{BASE}/musetalk_ok/bin/python",
    f"{BASE}/musetalk_ok/bin/python",
    f"{BASE}/musetalk_ok/bin/python3",
    f"{BASE}/musetalk_ok/bin/python3.11",
])

TTS_PY = TTS_PICK["path"]
MUSE_PY = MUSE_PICK["path"]


def _decode_b64(s: str) -> bytes:
    if not s:
        raise ValueError("b64 vacío")
    s = str(s).strip()
    if s.lower().startswith("data:") and "," in s:
        s = s.split(",", 1)[1].strip()
    s = "".join(s.split())
    s = s.replace("-", "+").replace("_", "/")
    pad = (-len(s)) % 4
    if pad:
        s += "=" * pad
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"b64 inválido: {e}")


def _b64_to_file(b64: str, out_path: str):
    raw = _decode_b64(b64)
    with open(out_path, "wb") as f:
        f.write(raw)


def _download_to_file(url: str, out_path: str):
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())


def _require_file(path: str, label: str):
    path = _norm_path(path)
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing {label}: {path}")
    return path


def _require_dir(path: str, label: str):
    path = _norm_path(path)
    if not os.path.isdir(path):
        raise RuntimeError(f"Missing {label}: {path}")
    return path


def _tts_make_wav(text: str, voice: str, lang: str, out_wav: str):
    speaker = FEMALE_REF_WAV if voice == "female" else MALE_REF_WAV
    _require_file(speaker, "speaker_wav")

    if not TTS_PICK["ok"]:
        raise RuntimeError(f"TTS python not runnable. Tested: {TTS_PICK['tested']}")

    cmd = [
        TTS_PY, "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker,
        "--out_wav", out_wav,
    ]
    _run(cmd)


def _musetalk_infer(input_mp4: str, audio_wav: str) -> str:
    root = _require_dir(MUSE_ROOT, "MUSE_ROOT (MuseTalk folder)")

    if not MUSE_PICK["ok"]:
        raise RuntimeError(f"MuseTalk python not runnable. Tested: {MUSE_PICK['tested']}")

    inputs_dir = os.path.join(root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    in_face = os.path.join(inputs_dir, "input_face.mp4")
    in_wav  = os.path.join(inputs_dir, "audio.wav")

    _run(["bash", "-lc", f"cp -f '{input_mp4}' '{in_face}'"])
    _run(["bash", "-lc", f"cp -f '{audio_wav}' '{in_wav}'"])

    cmd = [
        MUSE_PY, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
        "--bbox_shift", "0",
        "--use_float16",
    ]
    _run(cmd, cwd=root)

    results_dir = os.path.join(root, "results", "v15")
    cand = os.path.join(results_dir, "input_face_audio.mp4")
    if os.path.isfile(cand):
        return cand

    if not os.path.isdir(results_dir):
        raise RuntimeError(f"No results dir: {results_dir}")

    mp4s = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".mp4")]
    if not mp4s:
        raise RuntimeError("MuseTalk no produjo mp4")
    mp4s.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return mp4s[0]


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
    video_url = str(inp.get("video_url") or "").strip()
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
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = job.get("input") or {}
        mode = str(inp.get("mode") or inp.get("ping") or "").strip().lower()

        if mode in ("echo", "debug"):
            env_dump = {k: v for (k, v) in os.environ.items()
                        if k.startswith(("MUSE","TTS","BASE","FEMALE","MALE","RUNPOD","VOLUME","VOICES"))}

            return {
                "ok": True,
                "msg": "ECHO_OK",
                "base": BASE,
                "paths": {
                    "MUSE_ROOT": MUSE_ROOT,
                    "VOICES_DIR": VOICES_DIR,
                    "FEMALE_REF_WAV": FEMALE_REF_WAV,
                    "MALE_REF_WAV": MALE_REF_WAV,
                    "TTS_PY": TTS_PY,
                    "TTS_BIN": TTS_BIN,
                    "MUSE_PY": MUSE_PY,
                },
                "checks": {
                    "base_exists": _exists_dir(BASE),
                    "muse_root_exists": _exists_dir(MUSE_ROOT),
                    "voices_dir_exists": _exists_dir(VOICES_DIR),
                    "female_ref_exists": os.path.isfile(FEMALE_REF_WAV),
                    "male_ref_exists": os.path.isfile(MALE_REF_WAV),
                    # IMPORTANT: check runnable, not only isfile
                    "tts_py_runnable": bool(TTS_PICK["ok"]),
                    "muse_py_runnable": bool(MUSE_PICK["ok"]),
                    "tts_py_tested": TTS_PICK["tested"],
                    "muse_py_tested": MUSE_PICK["tested"],
                    "path_env": os.environ.get("PATH", ""),
                },
                "env_dump": env_dump,
            }

        if mode in ("ls", "list"):
            return {
                "ok": True,
                "base": BASE,
                "want": {
                    "TTS_PY": _norm_path(os.environ.get("TTS_PY")) or f"{BASE}/xtts_env/bin/python",
                    "MUSE_PY": _norm_path(os.environ.get("MUSE_PY")) or f"{BASE}/musetalk_ok/bin/python",
                },
                "picked": {
                    "TTS_PY": TTS_PY,
                    "MUSE_PY": MUSE_PY,
                    "TTS_PICK": TTS_PICK,
                    "MUSE_PICK": MUSE_PICK,
                },
                "list": {
                    "xtts_bin": _safe(["bash","-lc", f"ls -la {BASE}/xtts_env/bin | head -n 120"], timeout=30),
                    "musetalk_bin": _safe(["bash","-lc", f"ls -la {BASE}/musetalk_ok/bin | head -n 120"], timeout=30),
                    "python_links": _safe(["bash","-lc",
                        f"ls -la {BASE}/xtts_env/bin/python {BASE}/xtts_env/bin/python3 {BASE}/xtts_env/bin/python3.11 "
                        f"{BASE}/musetalk_ok/bin/python {BASE}/musetalk_ok/bin/python3 {BASE}/musetalk_ok/bin/python3.11 2>/dev/null || true"
                    ], timeout=30),
                }
            }

        if mode in ("voice_to_video", "voice2video", "v2v"):
            return voice_to_video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}. Use mode=echo|ls|voice_to_video"}

    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        _hard_cleanup()


runpod.serverless.start({"handler": handler})
