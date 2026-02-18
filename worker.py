# /app/worker.py
# RunPod Serverless Worker — IsabelaOS Voice2Video (XTTS -> MuseTalk)
# ✅ Stable: NO OpenMMLab installs inside container
# ✅ Uses container python3 (has cv2/diffusers) and just points MuseTalk repo via PYTHONPATH
# ✅ Modes:
#   - echo: environment/path checks
#   - voice2video: generate wav via XTTS (or accept audio_url) and run MuseTalk on a video_url
#   - muse_debug: runs quick python import checks inside MuseTalk repo context

import os
import re
import json
import time
import uuid
import base64
import shutil
import tempfile
import traceback
import subprocess
import urllib.request
from typing import Any, Dict, Optional, Tuple

import runpod

# ----------------------------
# ENV / Defaults
# ----------------------------
RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
COQUI_TOS_AGREED = os.environ.get("COQUI_TOS_AGREED", "1")
TTS_USE_GPU = os.environ.get("TTS_USE_GPU", "1").strip() not in ("0", "false", "False")

# Paths on volume
VOICES_DIR = os.path.join(RUNPOD_VOLUME_PATH, "voices")
FEMALE_REF_WAV = os.path.join(VOICES_DIR, "female_ref.wav")
MALE_REF_WAV = os.path.join(VOICES_DIR, "male_ref.wav")

# MuseTalk repo candidates (yours is here)
MUSE_REPO_CANDIDATES = [
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"),
    os.path.join(RUNPOD_VOLUME_PATH, "MuseTalk"),
]

# Some users have multiple configs; yours exists here:
MUSE_CONFIG_CANDIDATES = [
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk", "inference_config.json"),
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk", "inference_config.json."),
    os.path.join(RUNPOD_VOLUME_PATH, "inference_config.json"),
    os.path.join(RUNPOD_VOLUME_PATH, "inference_config.json."),
]

# Request time safety (RunPod execution timeout also exists)
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))  # keep below 600

# ----------------------------
# Helpers
# ----------------------------
def _now() -> float:
    return time.time()

def _tail(s: str, n: int = 1200) -> str:
    if not s:
        return ""
    return s[-n:]

def _clean_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Start from current env but ensure consistent flags.
    """
    e = dict(os.environ)
    e["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    e["TOKENIZERS_PARALLELISM"] = "false"
    e["COQUI_TOS_AGREED"] = str(COQUI_TOS_AGREED)
    if extra:
        e.update(extra)
    return e

def _run(cmd, cwd=None, env=None, timeout=HARD_TIMEOUT_SEC) -> Tuple[int, str]:
    """
    Run a command and capture stdout+stderr combined.
    """
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
        # drain
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
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)

def _looks_like_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _pick_musetalk_repo() -> Optional[str]:
    for p in MUSE_REPO_CANDIDATES:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "scripts", "inference.py")):
            return p
    # fallback: scan a bit
    for root, dirs, files in os.walk(RUNPOD_VOLUME_PATH):
        if "scripts" in dirs:
            inf = os.path.join(root, "scripts", "inference.py")
            if os.path.isfile(inf) and "MuseTalk" in root:
                return root
    return None

def _pick_musetalk_config(repo_root: str) -> Optional[str]:
    # Prefer config inside repo
    inside = os.path.join(repo_root, "inference_config.json")
    if os.path.isfile(inside):
        return inside
    inside_dot = inside + "."
    if os.path.isfile(inside_dot):
        return inside_dot

    for p in MUSE_CONFIG_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ----------------------------
# XTTS generator (calls /app/tts_generate.py)
# ----------------------------
def _tts_make_wav(text: str, lang: str, voice: str, out_wav: str) -> Dict[str, Any]:
    """
    Uses /app/tts_generate.py.
    Adds GPU fallback: if CUDA ECC error occurs, retries on CPU.
    """
    if voice == "male":
        speaker_wav = MALE_REF_WAV
    else:
        speaker_wav = FEMALE_REF_WAV

    if not os.path.isfile(speaker_wav):
        raise RuntimeError(f"speaker_wav not found: {speaker_wav}")

    tmpdir = os.path.dirname(out_wav)
    _safe_mkdir(tmpdir)

    # Try GPU (if enabled)
    def _call(use_gpu: bool) -> Tuple[int, str]:
        env = _clean_env({"TTS_USE_GPU": "1" if use_gpu else "0"})
        cmd = [
            "python3",
            "-u",
            "/app/tts_generate.py",
            "--text",
            text,
            "--lang",
            lang,
            "--speaker_wav",
            speaker_wav,
            "--out_wav",
            out_wav,
        ]
        return _run(cmd, env=env, timeout=HARD_TIMEOUT_SEC)

    tried = []
    if TTS_USE_GPU:
        code, out = _call(True)
        tried.append(("gpu", code, out))
        if code == 0 and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 1024:
            return {"ok": True, "device": "gpu", "log_tail": _tail(out)}
        # ECC/ CUDA failure -> fallback CPU
        if "uncorrectable ECC" in out or "CUDA error" in out:
            code2, out2 = _call(False)
            tried.append(("cpu", code2, out2))
            if code2 == 0 and os.path.isfile(out_wav) and os.path.getsize(out_wav) > 1024:
                return {"ok": True, "device": "cpu_fallback", "log_tail": _tail(out2)}
            raise RuntimeError("TTS failed (gpu then cpu)\n" + _tail(out2))
        raise RuntimeError("TTS failed (gpu)\n" + _tail(out))
    else:
        code, out = _call(False)
        tried.append(("cpu", code, out))
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
    """
    Runs: python3 -u scripts/inference.py ... (WITH PYTHONPATH=repo_root so 'musetalk' imports work)
    """
    if not os.path.isfile(os.path.join(repo_root, "scripts", "inference.py")):
        raise RuntimeError(f"MuseTalk inference.py not found in repo_root: {repo_root}")
    if not os.path.isfile(config_path):
        raise RuntimeError(f"MuseTalk config not found: {config_path}")
    if not os.path.isfile(input_mp4):
        raise RuntimeError(f"input_mp4 not found: {input_mp4}")
    if not os.path.isfile(audio_wav):
        raise RuntimeError(f"audio_wav not found: {audio_wav}")

    # MuseTalk scripts usually expect config file name relative to cwd.
    # We'll copy config into repo_root as inference_config.json to avoid path issues.
    local_cfg = os.path.join(repo_root, "inference_config.json")
    try:
        shutil.copyfile(config_path, local_cfg)
    except Exception:
        # if same file, ignore
        pass

    env = _clean_env(
        {
            # critical: make `import musetalk` work
            "PYTHONPATH": repo_root + (":" + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else ""),
        }
    )

    cmd = [
        "python3",
        "-u",
        "scripts/inference.py",
        "--inference_config",
        "inference_config.json",
        "--bbox_shift",
        str(int(bbox_shift)),
    ]
    if use_float16:
        cmd.append("--use_float16")

    code, out = _run(cmd, cwd=repo_root, env=env, timeout=HARD_TIMEOUT_SEC)

    if code != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

    # Best-effort: find an output mp4 in repo_root/results or outputs folder.
    # MuseTalk varies by fork; we scan for newest mp4.
    newest = None
    newest_mtime = 0.0
    for root, dirs, files in os.walk(repo_root):
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
        "log_tail": _tail(out),
        "output_mp4_guess": newest,
        "repo_root": repo_root,
    }

# ----------------------------
# Modes
# ----------------------------
def mode_echo() -> Dict[str, Any]:
    repo_root = _pick_musetalk_repo()
    cfg = _pick_musetalk_config(repo_root) if repo_root else None

    checks = {
        "voices_dir_exists": os.path.isdir(VOICES_DIR),
        "female_ref_exists": os.path.isfile(FEMALE_REF_WAV),
        "male_ref_exists": os.path.isfile(MALE_REF_WAV),
        "muse_repo_exists": bool(repo_root and os.path.isdir(repo_root)),
        "muse_scripts_inference_exists": bool(repo_root and os.path.isfile(os.path.join(repo_root, "scripts", "inference.py"))),
        "muse_config_exists": bool(cfg and os.path.isfile(cfg)),
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
        "python": shutil.which("python3") or "python3",
    }

def mode_muse_debug() -> Dict[str, Any]:
    repo_root = _pick_musetalk_repo()
    if not repo_root:
        raise RuntimeError("MuseTalk repo not found on volume")

    env = _clean_env({"PYTHONPATH": repo_root})
    # quick import check (will fail fast if cv2 missing, etc.)
    code, out = _run(
        ["python3", "-c", "import cv2; import sys; import importlib; import musetalk; print('OK', sys.version)"],
        cwd=repo_root,
        env=env,
        timeout=90,
    )
    if code != 0:
        raise RuntimeError("muse_debug failed\n" + _tail(out))

    return {"ok": True, "msg": "MUSE_DEBUG_OK", "repo_root": repo_root, "log_tail": _tail(out)}

def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Required:
      - video_url (mp4 public url) OR video_b64
    Optional:
      - text + lang + voice  (to run XTTS)
      - audio_url (wav url)  (skip XTTS)
      - bbox_shift (int)
      - use_float16 (bool)
    """
    start = _now()
    repo_root = _pick_musetalk_repo()
    if not repo_root:
        raise RuntimeError("MuseTalk repo not found on volume")
    cfg = _pick_musetalk_config(repo_root)
    if not cfg:
        raise RuntimeError("MuseTalk inference_config.json not found")

    bbox_shift = int(inp.get("bbox_shift", 0))
    use_float16 = bool(inp.get("use_float16", True))

    # temp workspace
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
            "output_mp4_guess": info.get("output_mp4_guess"),
            "log_tail": info.get("log_tail"),
        }
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

# ----------------------------
# Main handler
# ----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod sends: {"id": "...", "input": {...}}
    """
    try:
        inp = event.get("input") if isinstance(event, dict) else None
        if not isinstance(inp, dict):
            return {"ok": False, "error": "Missing or invalid input (expected JSON with field 'input')"}

        mode = str(inp.get("mode", "voice2video")).strip().lower()

        if mode == "echo":
            return mode_echo()
        if mode == "muse_debug":
            return mode_muse_debug()
        if mode in ("voice2video", "voice_to_video", "v2v"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "trace": traceback.format_exc(),
        }

# IMPORTANT: Serverless start (fixes "missing field(s): id or input" issues if used correctly by RunPod)
runpod.serverless.start({"handler": handler})
