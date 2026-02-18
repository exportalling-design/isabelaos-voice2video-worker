import os
import re
import json
import uuid
import time
import base64
import shutil
import traceback
import tempfile
import subprocess
from typing import Dict, Any, Optional

import requests
import runpod


# ----------------------------
# Config
# ----------------------------
RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
VOICES_DIR = os.path.join(RUNPOD_VOLUME_PATH, "voices")
FEMALE_REF_WAV = os.path.join(VOICES_DIR, "female_ref.wav")
MALE_REF_WAV = os.path.join(VOICES_DIR, "male_ref.wav")

# MuseTalk (está en el volumen)
MUSE_REPO_CANDIDATES = [
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"),
    os.path.join(RUNPOD_VOLUME_PATH, "MuseTalk"),
]
MUSE_CONFIG_CANDIDATES = [
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk", "inference_config.json"),
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk", "inference_config.json."),
    os.path.join(RUNPOD_VOLUME_PATH, "inference_config.json"),
    os.path.join(RUNPOD_VOLUME_PATH, "inference_config.json."),
]

# Timeout para infer (ojo serverless)
MUSE_TIMEOUT_SEC = int(os.environ.get("MUSE_TIMEOUT_SEC", "1200"))  # 20 min default


# ----------------------------
# Helpers
# ----------------------------
def _tail(s: str, n: int = 3000) -> str:
    if not s:
        return ""
    return s[-n:]


def _run(cmd, cwd=None, env=None, timeout=None) -> str:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    out = p.stdout or ""
    if p.returncode != 0:
        raise RuntimeError(f"CMD_FAILED: {' '.join(cmd)}\n{_tail(out)}\n")
    return out


def _download(url: str, out_path: str) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _pick_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _pick_musetalk_repo() -> Optional[str]:
    for p in MUSE_REPO_CANDIDATES:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "scripts", "inference.py")):
            return p
    # fallback: search quickly in volume_old
    vol_old = os.path.join(RUNPOD_VOLUME_PATH, "volume_old")
    if os.path.exists(vol_old):
        for root, dirs, files in os.walk(vol_old):
            if root.endswith("/MuseTalk") or root.endswith("\\MuseTalk"):
                if os.path.exists(os.path.join(root, "scripts", "inference.py")):
                    return root
    return None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _deep_replace_paths(obj: Any, replacements: Dict[str, str]) -> Any:
    # Reemplaza strings que contengan ciertos substrings
    if isinstance(obj, str):
        s = obj
        for k, v in replacements.items():
            if k in s:
                s = s.replace(k, v)
        return s
    if isinstance(obj, list):
        return [_deep_replace_paths(x, replacements) for x in obj]
    if isinstance(obj, dict):
        return {k: _deep_replace_paths(v, replacements) for k, v in obj.items()}
    return obj


# ----------------------------
# Modes
# ----------------------------
def mode_echo() -> Dict[str, Any]:
    repo = _pick_musetalk_repo()
    cfg = _pick_first_existing(MUSE_CONFIG_CANDIDATES)

    checks = {
        "voices_dir_exists": os.path.exists(VOICES_DIR),
        "female_ref_exists": os.path.exists(FEMALE_REF_WAV),
        "male_ref_exists": os.path.exists(MALE_REF_WAV),
        "muse_repo_exists": bool(repo),
        "muse_config_exists": bool(cfg),
        "muse_scripts_inference_exists": bool(repo and os.path.exists(os.path.join(repo, "scripts", "inference.py"))),
    }

    return {
        "ok": True,
        "msg": "ECHO_OK",
        "base": RUNPOD_VOLUME_PATH,
        "checks": checks,
        "env": {
            "RUNPOD_VOLUME_PATH": RUNPOD_VOLUME_PATH,
            "COQUI_TOS_AGREED": os.environ.get("COQUI_TOS_AGREED", ""),
            "TTS_USE_GPU": os.environ.get("TTS_USE_GPU", ""),
        },
        "paths": {
            "VOICES_DIR": VOICES_DIR,
            "FEMALE_REF_WAV": FEMALE_REF_WAV,
            "MALE_REF_WAV": MALE_REF_WAV,
            "MUSE_REPO_PICKED": repo,
            "MUSE_CONFIG_PICKED": cfg,
        },
        "python": shutil.which("python3") or "/usr/local/bin/python3",
    }


def mode_muse_debug() -> Dict[str, Any]:
    repo = _pick_musetalk_repo()
    cfg = _pick_first_existing(MUSE_CONFIG_CANDIDATES)
    if not repo or not cfg:
        return {"ok": False, "error": "MuseTalk repo/config not found", "repo": repo, "cfg": cfg}

    # solo probar imports básicos + existence
    # (no corre inference)
    try:
        import cv2  # noqa
        import diffusers  # noqa
        import mmpose  # noqa
    except Exception as e:
        return {"ok": False, "error": "Python deps missing", "trace": traceback.format_exc()}

    return {
        "ok": True,
        "msg": "MUSE_DEBUG_OK",
        "repo": repo,
        "config": cfg,
        "python": shutil.which("python3") or "/usr/local/bin/python3",
    }


def _tts_make_wav(text: str, lang: str, voice: str, out_wav: str) -> Dict[str, Any]:
    if voice == "male":
        speaker_wav = MALE_REF_WAV
    else:
        speaker_wav = FEMALE_REF_WAV

    if not os.path.exists(speaker_wav):
        raise RuntimeError(f"Missing speaker_wav: {speaker_wav}")

    cmd = [
        "python3", "-u", "/app/tts_generate.py",
        "--text", text,
        "--lang", lang,
        "--speaker_wav", speaker_wav,
        "--out_wav", out_wav,
    ]
    out = _run(cmd, env=os.environ.copy(), timeout=1200)
    return {"ok": True, "speaker_wav": speaker_wav, "out": _tail(out)}


def _musetalk_infer(repo_root: str, base_config_path: str, input_mp4: str, audio_wav: str) -> Dict[str, Any]:
    # Armamos run dir dentro del repo para que el config sea relativo
    run_id = str(uuid.uuid4())[:8]
    run_dir = os.path.join(repo_root, "inputs", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    local_video = os.path.join(run_dir, "input.mp4")
    local_audio = os.path.join(run_dir, "input.wav")
    shutil.copyfile(input_mp4, local_video)
    shutil.copyfile(audio_wav, local_audio)

    # Cargar config base y tratar de apuntar a nuestro run_dir
    cfg = _load_json(base_config_path)

    # Intento “best effort” de reemplazo:
    # Si el config tiene rutas dentro de "inputs/", lo redirigimos a inputs/run_xxx/
    rel_run_dir = os.path.relpath(run_dir, repo_root).replace("\\", "/")  # "inputs/run_xxx"
    replacements = {
        "inputs/": f"{rel_run_dir}/",
        "inputs\\": f"{rel_run_dir}/",
    }
    cfg2 = _deep_replace_paths(cfg, replacements)

    # además: si hay keys típicas, forzamos
    def force_key(d: Dict[str, Any], keys, value):
        for k in keys:
            if k in d:
                d[k] = value

    # puede estar anidado; intentamos en raíz y subdicts comunes
    for d in [cfg2] + [v for v in cfg2.values() if isinstance(v, dict)]:
        force_key(d, ["video_path", "source_video", "video", "input_video"], local_video)
        force_key(d, ["audio_path", "driving_audio", "audio", "input_audio"], local_audio)

    tmp_cfg_path = os.path.join(run_dir, "inference_config.json")
    _save_json(tmp_cfg_path, cfg2)

    # correr inference usando python3 del contenedor
    env = os.environ.copy()
    # clave: para que "import musetalk" funcione desde el repo
    env["PYTHONPATH"] = repo_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [
        "python3", "-u", "scripts/inference.py",
        "--inference_config", os.path.relpath(tmp_cfg_path, repo_root).replace("\\", "/"),
        "--bbox_shift", "0",
        "--use_float16",
    ]

    out = _run(cmd, cwd=repo_root, env=env, timeout=MUSE_TIMEOUT_SEC)

    # Buscar mp4 output reciente
    candidates = []
    for root, dirs, files in os.walk(repo_root):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                p = os.path.join(root, fn)
                try:
                    candidates.append((os.path.getmtime(p), p))
                except Exception:
                    pass
    candidates.sort(reverse=True)
    out_mp4 = candidates[0][1] if candidates else None
    if not out_mp4:
        raise RuntimeError("MuseTalk ran but no mp4 found.\n" + _tail(out))

    return {
        "ok": True,
        "run_dir": run_dir,
        "out_mp4_path": out_mp4,
        "log_tail": _tail(out),
    }


def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    """
    input esperado (mínimo):
    {
      "mode": "voice2video",
      "video_url": "https://....mp4",
      "text": "Hola ...",           (opcional si das audio_url)
      "audio_url": "https://....wav" (opcional si usas TTS)
      "voice": "female|male",
      "lang": "es"
    }
    """
    repo = _pick_musetalk_repo()
    cfg = _pick_first_existing(MUSE_CONFIG_CANDIDATES)
    if not repo or not cfg:
        return {"ok": False, "error": "MuseTalk repo/config not found", "repo": repo, "cfg": cfg}

    video_url = inp.get("video_url")
    audio_url = inp.get("audio_url")
    text = inp.get("text", "")
    voice = inp.get("voice", "female")
    lang = inp.get("lang", "es")

    if not video_url:
        return {"ok": False, "error": "Missing video_url"}

    tmpdir = tempfile.mkdtemp(prefix="v2v_")
    in_mp4 = os.path.join(tmpdir, "input.mp4")
    _download(video_url, in_mp4)

    tts_wav = os.path.join(tmpdir, "audio.wav")
    tts_info = None

    if audio_url:
        _download(audio_url, tts_wav)
    else:
        if not text.strip():
            return {"ok": False, "error": "Provide audio_url or text"}
        tts_info = _tts_make_wav(text=text, lang=lang, voice=voice, out_wav=tts_wav)

    musetalk_info = _musetalk_infer(repo_root=repo, base_config_path=cfg, input_mp4=in_mp4, audio_wav=tts_wav)

    # devolver base64 (si tu backend luego lo sube a Supabase)
    out_mp4_path = musetalk_info["out_mp4_path"]
    with open(out_mp4_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "ok": True,
        "mode": "voice2video",
        "tts": tts_info,
        "musetalk": {k: musetalk_info[k] for k in ["ok", "out_mp4_path", "run_dir"]},
        "out_mp4_b64": b64,
        "out_filename": os.path.basename(out_mp4_path),
    }


# ----------------------------
# RunPod handler
# ----------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless manda:
      { "input": {...}, "id": "...", ... }
    Si te mandan mal el JSON, verás: "missing field(s): id or input"
    """
    try:
        if not isinstance(event, dict) or "input" not in event:
            return {"ok": False, "error": "Invalid event: missing input", "event_keys": list(event.keys()) if isinstance(event, dict) else str(type(event))}

        inp = event.get("input") or {}
        mode = (inp.get("mode") or "echo").strip()

        if mode == "echo":
            return mode_echo()
        if mode == "muse_debug":
            return mode_muse_debug()
        if mode in ("voice2video", "voice_to_video"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}

    except Exception:
        return {"ok": False, "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
