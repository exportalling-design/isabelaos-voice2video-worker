# /app/worker.py
# RunPod Serverless Worker — IsabelaOS Voice2Video (XTTS -> MuseTalk)
# ✅ NO recrea venv
# ✅ Escanea /runpod-volume y elige automáticamente el python que ya tenga:
#    cv2 + mmcv + mmengine + mmdet + mmpose + musetalk
# ✅ Modes: scan, echo, voice2video

import os
import json
import time
import base64
import shutil
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, Tuple, List

import runpod

# ----------------------------
# ENV / Defaults
# ----------------------------
RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
COQUI_TOS_AGREED = os.environ.get("COQUI_TOS_AGREED", "1")
TTS_USE_GPU = os.environ.get("TTS_USE_GPU", "1").strip().lower() not in ("0", "false", "no")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))  # < 600
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "25"))   # per candidate

# Paths on volume
VOICES_DIR = os.path.join(RUNPOD_VOLUME_PATH, "voices")
FEMALE_REF_WAV = os.path.join(VOICES_DIR, "female_ref.wav")
MALE_REF_WAV = os.path.join(VOICES_DIR, "male_ref.wav")

MUSE_REPO_CANDIDATES = [
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"),
    os.path.join(RUNPOD_VOLUME_PATH, "MuseTalk"),
]

MUSE_CONFIG_CANDIDATES = [
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk", "inference_config.json"),
    os.path.join(RUNPOD_VOLUME_PATH, "MuseTalk", "inference_config.json"),
    os.path.join(RUNPOD_VOLUME_PATH, "inference_config.json"),
]

# Cache (para no escanear cada request)
_SELECTED_PY: Optional[str] = None
_SELECTED_REASON: Optional[Dict[str, Any]] = None


# ----------------------------
# Helpers
# ----------------------------
def _now() -> float:
    return time.time()

def _tail(s: str, n: int = 1600) -> str:
    return (s or "")[-n:]

def _clean_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    e = dict(os.environ)
    e["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    e["TOKENIZERS_PARALLELISM"] = "false"
    e["COQUI_TOS_AGREED"] = str(COQUI_TOS_AGREED)
    if extra:
        e.update(extra)
    return e

def _run(cmd, cwd=None, env=None, timeout=HARD_TIMEOUT_SEC) -> Tuple[int, str]:
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    out_lines = []
    start = _now()
    try:
        while True:
            if p.poll() is not None:
                break
            line = p.stdout.readline()
            if line:
                out_lines.append(line)
            if _now() - start > timeout:
                try:
                    p.kill()
                except Exception:
                    pass
                out_lines.append("\n[TIMEOUT] killed process\n")
                return 124, "".join(out_lines)
        rest = p.stdout.read()
        if rest:
            out_lines.append(rest)
        return p.returncode or 0, "".join(out_lines)
    except Exception as ex:
        try:
            p.kill()
        except Exception:
            pass
        out_lines.append(f"\n[EXCEPTION] {ex}\n")
        return 1, "".join(out_lines)

def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)

def _looks_like_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _pick_musetalk_repo() -> Optional[str]:
    for p in MUSE_REPO_CANDIDATES:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "scripts", "inference.py")):
            return p
    # fallback scan
    for root, dirs, files in os.walk(RUNPOD_VOLUME_PATH):
        if "scripts" in dirs:
            inf = os.path.join(root, "scripts", "inference.py")
            if os.path.isfile(inf) and "MuseTalk" in root:
                return root
    return None

def _pick_musetalk_config(repo_root: str) -> Optional[str]:
    inside = os.path.join(repo_root, "inference_config.json")
    if os.path.isfile(inside):
        return inside
    for p in MUSE_CONFIG_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


# ----------------------------
# 🔎 Python/Venv auto-detect in volume
# ----------------------------
def _list_candidate_pythons() -> List[str]:
    """
    Busca bin/python y bin/python3 dentro del volumen.
    No depende de 'find' externo.
    """
    candidates = []
    # Heurística: carpetas típicas
    likely_roots = [
        RUNPOD_VOLUME_PATH,
        os.path.join(RUNPOD_VOLUME_PATH, "volume_old"),
    ]
    seen = set()

    for root in likely_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # limit depth un poco (para no tardar siglos)
            rel = os.path.relpath(dirpath, root)
            depth = rel.count(os.sep)
            if depth > 6:
                dirnames[:] = []
                continue

            if dirpath.endswith(os.sep + "bin") or dirpath.endswith("/bin"):
                py1 = os.path.join(dirpath, "python")
                py3 = os.path.join(dirpath, "python3")
                for p in (py1, py3):
                    if os.path.isfile(p) and os.access(p, os.X_OK) and p not in seen:
                        seen.add(p)
                        candidates.append(p)

    # siempre incluir python3 del container como último fallback
    sys_py = shutil.which("python3") or "python3"
    if sys_py not in seen:
        candidates.append(sys_py)

    return candidates

def _probe_python(py: str, repo_root: Optional[str]) -> Dict[str, Any]:
    """
    Prueba imports críticos. Si repo_root existe, setea PYTHONPATH=repo_root.
    """
    env = _clean_env({})
    if repo_root:
        env["PYTHONPATH"] = repo_root + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

    # OJO: no todos tienen mmdet, pero MuseTalk normalmente necesita mmpose + mmcv + mmengine + cv2
    # Hacemos 2 probes: mínimo y completo.
    code1, out1 = _run(
        [py, "-c", "import sys; import cv2; print('OK_cv2', sys.version)"],
        cwd=repo_root or None,
        env=env,
        timeout=SCAN_TIMEOUT_SEC,
    )
    ok_cv2 = (code1 == 0)

    code2, out2 = _run(
        [py, "-c", "import mmcv, mmengine; import mmpose; print('OK_openmmlab')"],
        cwd=repo_root or None,
        env=env,
        timeout=SCAN_TIMEOUT_SEC,
    )
    ok_openmmlab = (code2 == 0)

    code3, out3 = _run(
        [py, "-c", "import musetalk; print('OK_musetalk')"],
        cwd=repo_root or None,
        env=env,
        timeout=SCAN_TIMEOUT_SEC,
    )
    ok_musetalk = (code3 == 0)

    return {
        "py": py,
        "ok_cv2": ok_cv2,
        "ok_openmmlab": ok_openmmlab,
        "ok_musetalk": ok_musetalk,
        "tail_cv2": _tail(out1),
        "tail_openmmlab": _tail(out2),
        "tail_musetalk": _tail(out3),
    }

def _select_best_python(repo_root: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Selección:
    1) cv2 + openmmlab + musetalk ✅
    2) cv2 + openmmlab ✅ (musetalk puede venir por PYTHONPATH pero import falla si no ve repo)
    3) cv2 ✅
    4) fallback python3 del container
    """
    global _SELECTED_PY, _SELECTED_REASON
    if _SELECTED_PY and _SELECTED_REASON:
        return _SELECTED_PY, _SELECTED_REASON

    cands = _list_candidate_pythons()
    results = []
    for py in cands:
        r = _probe_python(py, repo_root)
        results.append(r)

    # scoring
    def score(r):
        return (3 if (r["ok_cv2"] and r["ok_openmmlab"] and r["ok_musetalk"]) else
                2 if (r["ok_cv2"] and r["ok_openmmlab"]) else
                1 if r["ok_cv2"] else
                0)

    results_sorted = sorted(results, key=score, reverse=True)
    best = results_sorted[0]
    _SELECTED_PY = best["py"]
    _SELECTED_REASON = {"best": best, "all": results_sorted[:12]}
    return _SELECTED_PY, _SELECTED_REASON


# ----------------------------
# XTTS generator (calls /app/tts_generate.py)
# ----------------------------
def _tts_make_wav(text: str, lang: str, voice: str, out_wav: str) -> Dict[str, Any]:
    speaker_wav = MALE_REF_WAV if voice == "male" else FEMALE_REF_WAV
    if not os.path.isfile(speaker_wav):
        raise RuntimeError(f"speaker_wav not found: {speaker_wav}")

    _safe_mkdir(os.path.dirname(out_wav))

    def _call(use_gpu: bool) -> Tuple[int, str]:
        env = _clean_env({"TTS_USE_GPU": "1" if use_gpu else "0"})
        cmd = [
            "python3", "-u", "/app/tts_generate.py",
            "--text", text,
            "--lang", lang,
            "--speaker_wav", speaker_wav,
            "--out_wav", out_wav,
        ]
        return _run(cmd, env=env, timeout=HARD_TIMEOUT_SEC)

    if TTS_USE_GPU:
        code, out = _call(True)
        if code == 0 and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 1024:
            return {"ok": True, "device": "gpu", "log_tail": _tail(out)}
        low = (out or "").lower()
        if ("ecc" in low) or ("cuda error" in low) or ("device-side assert" in low):
            code2, out2 = _call(False)
            if code2 == 0 and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 1024:
                return {"ok": True, "device": "cpu_fallback", "log_tail": _tail(out2)}
            raise RuntimeError("TTS failed (gpu then cpu)\n" + _tail(out2))
        raise RuntimeError("TTS failed (gpu)\n" + _tail(out))
    else:
        code, out = _call(False)
        if code == 0 and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 1024:
            return {"ok": True, "device": "cpu", "log_tail": _tail(out)}
        raise RuntimeError("TTS failed (cpu)\n" + _tail(out))


# ----------------------------
# MuseTalk runner
# ----------------------------
def _musetalk_infer(
    repo_root: str,
    config_path: str,
    input_mp4: str,
    audio_wav: str,
    bbox_shift: int = 0,
    use_float16: bool = True,
) -> Dict[str, Any]:

    py, reason = _select_best_python(repo_root)

    if not os.path.isfile(os.path.join(repo_root, "scripts", "inference.py")):
        raise RuntimeError(f"MuseTalk inference.py not found in repo_root: {repo_root}")
    if not os.path.isfile(config_path):
        raise RuntimeError(f"MuseTalk config not found: {config_path}")
    if not os.path.isfile(input_mp4):
        raise RuntimeError(f"input_mp4 not found: {input_mp4}")
    if not os.path.isfile(audio_wav):
        raise RuntimeError(f"audio_wav not found: {audio_wav}")

    # Copia config al repo para que inference lo vea simple
    local_cfg = os.path.join(repo_root, "inference_config.json")
    try:
        shutil.copyfile(config_path, local_cfg)
    except Exception:
        pass

    # ⚠️ MuseTalk muchas veces toma input/audio desde el inference_config.json
    # Si tu config ya apunta a rutas internas del repo, ok.
    # Si necesitás inyectar input/audio, eso se hace editando el JSON (lo dejamos igual por ahora).

    env = _clean_env({
        "PYTHONPATH": repo_root + (":" + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")
    })

    cmd = [
        py, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
        "--bbox_shift", str(int(bbox_shift)),
    ]
    if use_float16:
        cmd.append("--use_float16")

    code, out = _run(cmd, cwd=repo_root, env=env, timeout=HARD_TIMEOUT_SEC)
    if code != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

    # busca mp4 más nuevo dentro del repo (best-effort)
    newest = None
    newest_mtime = 0.0
    for root, _, files in os.walk(repo_root):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                fp = os.path.join(root, fn)
                try:
                    mt = os.path.getmtime(fp)
                    if mt > newest_mtime:
                        newest_mtime = mt
                        newest = fp
                except Exception:
                    pass

    return {
        "ok": True,
        "python_used": py,
        "python_probe_best": reason.get("best"),
        "log_tail": _tail(out),
        "output_mp4_guess": newest,
    }


# ----------------------------
# Modes
# ----------------------------
def mode_scan() -> Dict[str, Any]:
    repo_root = _pick_musetalk_repo()
    py, reason = _select_best_python(repo_root)
    return {
        "ok": True,
        "msg": "SCAN_OK",
        "repo_root": repo_root,
        "picked_python": py,
        "probe_best": reason.get("best"),
        "top_candidates": reason.get("all"),
    }

def mode_echo() -> Dict[str, Any]:
    repo_root = _pick_musetalk_repo()
    cfg = _pick_musetalk_config(repo_root) if repo_root else None
    py, reason = _select_best_python(repo_root)

    checks = {
        "voices_dir_exists": os.path.isdir(VOICES_DIR),
        "female_ref_exists": os.path.isfile(FEMALE_REF_WAV),
        "male_ref_exists": os.path.isfile(MALE_REF_WAV),
        "muse_repo_exists": bool(repo_root and os.path.isdir(repo_root)),
        "muse_scripts_inference_exists": bool(repo_root and os.path.isfile(os.path.join(repo_root, "scripts", "inference.py"))),
        "muse_config_exists": bool(cfg and os.path.isfile(cfg)),
        "picked_python": py,
        "probe_best": reason.get("best"),
    }

    return {
        "ok": True,
        "msg": "ECHO_OK",
        "base": RUNPOD_VOLUME_PATH,
        "env": {
            "COQUI_TOS_AGREED": COQUI_TOS_AGREED,
            "RUNPOD_VOLUME_PATH": RUNPOD_VOLUME_PATH,
            "TTS_USE_GPU": "1" if TTS_USE_GPU else "0",
        },
        "checks": checks,
        "paths": {
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "MUSE_REPO_PICKED": repo_root,
            "MUSE_CONFIG_PICKED": cfg,
        },
    }

def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    start = _now()
    repo_root = _pick_musetalk_repo()
    if not repo_root:
        raise RuntimeError("MuseTalk repo not found on volume")
    cfg = _pick_musetalk_config(repo_root)
    if not cfg:
        raise RuntimeError("MuseTalk inference_config.json not found")

    bbox_shift = int(inp.get("bbox_shift", 0))
    use_float16 = bool(inp.get("use_float16", True))

    tmp = tempfile.mkdtemp(prefix="v2v_")
    try:
        in_mp4 = os.path.join(tmp, "input.mp4")
        tts_wav = os.path.join(tmp, "tts.wav")

        # ---- get video ----
        if _looks_like_url(inp.get("video_url")):
            _download(inp["video_url"], in_mp4)
        elif isinstance(inp.get("video_b64"), str) and inp["video_b64"]:
            raw = base64.b64decode(inp["video_b64"])
            with open(in_mp4, "wb") as f:
                f.write(raw)
        else:
            raise RuntimeError("Missing video_url or video_b64")

        if not os.path.isfile(in_mp4) or os.path.getsize(in_mp4) < 1024:
            raise RuntimeError("Downloaded/decoded video is invalid or too small")

        # ---- get audio ----
        if _looks_like_url(inp.get("audio_url")):
            _download(inp["audio_url"], tts_wav)
        else:
            text = str(inp.get("text", "")).strip()
            if not text:
                raise RuntimeError("Missing audio_url OR text for XTTS")
            lang = str(inp.get("lang", "es")).strip() or "es"
            voice = str(inp.get("voice", "female")).strip().lower()
            if voice not in ("female", "male"):
                voice = "female"
            _tts_make_wav(text=text, lang=lang, voice=voice, out_wav=tts_wav)

        if not os.path.isfile(tts_wav) or os.path.getsize(tts_wav) < 1024:
            raise RuntimeError("Audio wav is invalid or too small")

        # ---- run MuseTalk ----
        info = _musetalk_infer(
            repo_root=repo_root,
            config_path=cfg,
            input_mp4=in_mp4,
            audio_wav=tts_wav,
            bbox_shift=bbox_shift,
            use_float16=use_float16,
        )

        elapsed = int((_now() - start) * 1000)
        return {
            "ok": True,
            "msg": "VOICE2VIDEO_OK",
            "execution_ms": elapsed,
            "repo_root": repo_root,
            "config_used": cfg,
            "python_used": info.get("python_used"),
            "python_probe_best": info.get("python_probe_best"),
            "output_mp4_guess": info.get("output_mp4_guess"),
            "log_tail": info.get("log_tail"),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------
# Main handler
# ----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = event.get("input") if isinstance(event, dict) else None
        if not isinstance(inp, dict):
            return {"ok": False, "error": "Missing or invalid input (expected JSON with field 'input')"}

        mode = str(inp.get("mode", "voice2video")).strip().lower()

        if mode == "scan":
            return mode_scan()
        if mode == "echo":
            return mode_echo()
        if mode in ("voice2video", "voice_to_video", "v2v"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
