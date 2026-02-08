# /workspace/worker.py
# RunPod Serverless Worker: XTTS (text->audio) + MuseTalk (lipsync)
# - Expects /runpod-volume mounted with:
#   /runpod-volume/xtts_env/bin/tts
#   /runpod-volume/musetalk_ok/bin/python
#   /runpod-volume/MuseTalk/...

import os
import json
import base64
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import runpod


# ----------------------------
# ENV / Paths (defaults)
# ----------------------------
VOLUME = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")

XTTS_ENV = os.environ.get("XTTS_ENV_PATH", f"{VOLUME}/xtts_env")
MUSE_ENV = os.environ.get("MUSE_ENV_PATH", f"{VOLUME}/musetalk_ok")
MUSE_ROOT = os.environ.get("MUSE_ROOT", f"{VOLUME}/MuseTalk")

VOICES_DIR = os.environ.get("VOICES_DIR", f"{VOLUME}/voices")

DEFAULT_SPEAKER_WAV = os.environ.get("DEFAULT_SPEAKER_WAV", f"{VOICES_DIR}/female_ref.wav")
DEFAULT_LANG = os.environ.get("DEFAULT_LANG", "es")
DEFAULT_TTS_MODEL = os.environ.get("DEFAULT_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

# MuseTalk default I/O
MUSE_INPUT_VIDEO = os.environ.get("MUSE_INPUT_VIDEO", f"{MUSE_ROOT}/inputs/input_video.mp4")
MUSE_INPUT_AUDIO = os.environ.get("MUSE_INPUT_AUDIO", f"{MUSE_ROOT}/inputs/audio.wav")
MUSE_OUTPUT_DIR = os.environ.get("MUSE_OUTPUT_DIR", f"{MUSE_ROOT}/outputs")

# Return format: "b64" or "path"
DEFAULT_RETURN = os.environ.get("DEFAULT_RETURN", "path").lower()

# Cache dirs to avoid "Disk quota exceeded" on ephemeral FS
CACHE_ROOT = os.environ.get("CACHE_ROOT", f"{VOLUME}/cache")
os.environ.setdefault("XDG_CACHE_HOME", f"{CACHE_ROOT}/xdg")
os.environ.setdefault("HF_HOME", f"{CACHE_ROOT}/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", f"{CACHE_ROOT}/hf")
os.environ.setdefault("TORCH_HOME", f"{CACHE_ROOT}/torch")
os.environ.setdefault("NUMBA_CACHE_DIR", f"{CACHE_ROOT}/numba")
os.environ.setdefault("MPLCONFIGDIR", f"{CACHE_ROOT}/mpl")


def _json_error(msg: str, extra: Dict[str, Any] = None, code: str = "BAD_REQUEST") -> Dict[str, Any]:
    out = {"ok": False, "error": msg, "code": code}
    if extra:
        out["extra"] = extra
    return out


def _ensure_dirs():
    Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
    Path(MUSE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{MUSE_ROOT}/inputs").mkdir(parents=True, exist_ok=True)


def _exists_or_fail(p: str, name: str):
    if not Path(p).exists():
        raise FileNotFoundError(f"{name} not found: {p}")


def _run(cmd: list, cwd: str = None) -> Tuple[int, str]:
    """Run a command, return (exit_code, combined_output)."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    out_lines = []
    for line in proc.stdout:
        out_lines.append(line)
    proc.wait()
    return proc.returncode, "".join(out_lines)


def _maybe_write_b64_file(b64: str, path: str):
    raw = base64.b64decode(b64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(raw)


def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_tts(text: str, speaker_wav: str, lang: str, out_path: str, model_name: str) -> Dict[str, Any]:
    tts_bin = f"{XTTS_ENV}/bin/tts"
    _exists_or_fail(tts_bin, "XTTS binary (tts)")
    _exists_or_fail(speaker_wav, "speaker_wav")

    cmd = [
        tts_bin,
        "--text", text,
        "--model_name", model_name,
        "--speaker_wav", speaker_wav,
        "--language_idx", lang,
        "--out_path", out_path,
    ]

    code, logs = _run(cmd, cwd=str(Path(MUSE_ROOT).parent))
    if code != 0:
        return _json_error("XTTS failed", {"exit_code": code, "logs_tail": logs[-4000:]}, code="TTS_FAILED")

    if not Path(out_path).exists():
        return _json_error("XTTS reported success but output not found", {"out_path": out_path}, code="TTS_NO_OUTPUT")

    return {"ok": True, "audio_path": out_path, "logs_tail": logs[-2000:]}


def find_musetalk_entry() -> str:
    """
    Try to locate the MuseTalk inference entry.
    Adapt this if your repo uses a different file.
    """
    candidates = [
        f"{MUSE_ROOT}/inference.py",
        f"{MUSE_ROOT}/scripts/inference.py",
        f"{MUSE_ROOT}/app.py",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    # If none found, fail with directory listing hint
    raise FileNotFoundError(f"Could not find MuseTalk entry. Checked: {candidates}")


def run_musetalk(video_path: str, audio_path: str, out_dir: str) -> Dict[str, Any]:
    py = f"{MUSE_ENV}/bin/python"
    _exists_or_fail(py, "MuseTalk python")
    _exists_or_fail(video_path, "input video")
    _exists_or_fail(audio_path, "input audio")

    entry = find_musetalk_entry()

    # Common MuseTalk style args (you might adjust flags to match your repo)
    cmd = [
        py, entry,
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--result_dir", out_dir,
    ]

    code, logs = _run(cmd, cwd=MUSE_ROOT)
    if code != 0:
        return _json_error("MuseTalk failed", {"exit_code": code, "logs_tail": logs[-4000:]}, code="MUSETALK_FAILED")

    # Try to find a resulting mp4 (common patterns)
    out_dir_p = Path(out_dir)
    mp4s = sorted(out_dir_p.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        return _json_error("MuseTalk finished but no mp4 found in outputs", {"out_dir": out_dir}, code="MUSETALK_NO_OUTPUT")

    return {"ok": True, "video_path": str(mp4s[0]), "logs_tail": logs[-2000:]}


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected request:
    {
      "input": {
        "mode": "tts" | "lipsync" | "tts_lipsync",
        "text": "Fui creada por IsabelaOS",
        "speaker_wav_path": "/runpod-volume/voices/female_ref.wav",   (optional)
        "speaker": "female" | "male",                                 (optional)
        "lang": "es",                                                 (optional)
        "tts_model": "...xtts_v2",                                    (optional)
        "video_path": "/runpod-volume/MuseTalk/inputs/input_video.mp4",(optional)
        "video_b64": "...",                                           (optional)
        "audio_path": "/runpod-volume/MuseTalk/inputs/audio.wav",     (optional)
        "return": "b64" | "path"                                      (optional)
      }
    }
    """
    _ensure_dirs()

    inp = job.get("input", {}) or {}

    # ✅ VERIFICACIÓN REQUEST: esto te deja claro qué llegó al endpoint
    # (sale en logs del endpoint en RunPod)
    print("=== REQUEST_KEYS ===")
    print(sorted(list(inp.keys())))
    print("=== REQUEST_PREVIEW ===")
    safe_preview = {k: ("<b64>" if "b64" in k else inp.get(k)) for k in inp.keys()}
    print(json.dumps(safe_preview, ensure_ascii=False)[:2000])

    mode = (inp.get("mode") or "tts_lipsync").lower()
    ret = (inp.get("return") or DEFAULT_RETURN).lower()

    # Resolve speaker wav
    speaker = (inp.get("speaker") or "").lower().strip()
    speaker_wav = inp.get("speaker_wav_path") or DEFAULT_SPEAKER_WAV
    if speaker in ("female", "mujer"):
        speaker_wav = f"{VOICES_DIR}/female_ref.wav"
    elif speaker in ("male", "hombre"):
        speaker_wav = f"{VOICES_DIR}/male_ref.wav"

    lang = (inp.get("lang") or DEFAULT_LANG).strip()
    tts_model = (inp.get("tts_model") or DEFAULT_TTS_MODEL).strip()

    video_path = inp.get("video_path") or MUSE_INPUT_VIDEO
    audio_path = inp.get("audio_path") or MUSE_INPUT_AUDIO

    # If they send base64 video/audio, write to expected paths
    if inp.get("video_b64"):
        _maybe_write_b64_file(inp["video_b64"], video_path)
    if inp.get("audio_b64"):
        _maybe_write_b64_file(inp["audio_b64"], audio_path)

    out: Dict[str, Any] = {"ok": True, "mode": mode}

    try:
        # Validate volume presence
        if not Path(VOLUME).exists():
            return _json_error("Volume mount not found. Serverless must mount network volume to /runpod-volume.",
                               {"RUNPOD_VOLUME_PATH": VOLUME}, code="NO_VOLUME")

        if mode in ("tts", "tts_lipsync"):
            text = (inp.get("text") or "").strip()
            if not text:
                return _json_error("Missing 'text' for TTS", {"needed": ["text"]})

            tts_res = generate_tts(
                text=text,
                speaker_wav=speaker_wav,
                lang=lang,
                out_path=audio_path,
                model_name=tts_model,
            )
            if not tts_res.get("ok"):
                return tts_res
            out["tts"] = tts_res

        if mode in ("lipsync", "tts_lipsync"):
            lip_res = run_musetalk(
                video_path=video_path,
                audio_path=audio_path,
                out_dir=MUSE_OUTPUT_DIR,
            )
            if not lip_res.get("ok"):
                return lip_res
            out["musetalk"] = lip_res

        # Return files
        if ret == "b64":
            if out.get("musetalk", {}).get("video_path"):
                out["result_video_b64"] = _file_to_b64(out["musetalk"]["video_path"])
            if Path(audio_path).exists():
                out["result_audio_b64"] = _file_to_b64(audio_path)
        else:
            out["result_audio_path"] = audio_path
            if out.get("musetalk", {}).get("video_path"):
                out["result_video_path"] = out["musetalk"]["video_path"]

        return out

    except Exception as e:
        return _json_error("Unhandled exception in worker", {
            "message": str(e),
            "trace": traceback.format_exc()[-6000:],
        }, code="CRASH")


runpod.serverless.start({"handler": handler})
