# /app/worker.py
# RunPod Serverless Worker — IsabelaOS Voice2Video (MuseTalk)
# ✅ No toca /runpod-volume (solo lee repo/modelos y escribe outputs dentro del repo si MuseTalk lo hace)
# ✅ Usa python del container
# ✅ NO usa venv del volumen
# ✅ NO requiere mmpose

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

# --------------------------------------------------
# ENV
# --------------------------------------------------
RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
COQUI_TOS_AGREED = os.environ.get("COQUI_TOS_AGREED", "1")
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))  # < 600
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "20"))

# ✅ Fuerza python del container (endpoint)
PY_CONTAINER = os.environ.get("PY_CONTAINER", "/usr/local/bin/python3")

# MuseTalk repo candidates (según tus logs existe /runpod-volume/volume_old/MuseTalk)
MUSE_REPO_CANDIDATES = [
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"),
    os.path.join(RUNPOD_VOLUME_PATH, "MuseTalk"),
    os.path.join("/workspace", "volume_old", "MuseTalk"),
    os.path.join("/workspace", "MuseTalk"),
]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _now() -> float:
    return time.time()

def _tail(s: str, n: int = 2000) -> str:
    return (s or "")[-n:]

def _run(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
         timeout: int = HARD_TIMEOUT_SEC) -> Tuple[int, str]:
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
    out_lines: List[str] = []
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

def _clean_env(repo_root: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["COQUI_TOS_AGREED"] = str(COQUI_TOS_AGREED or "1")
    # MuseTalk como repo => PYTHONPATH
    if repo_root:
        env["PYTHONPATH"] = repo_root + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    return env

def _looks_like_url(s: Any) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)

def _b64_to_file(b64: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(base64.b64decode(b64))

def _pick_musetalk_repo() -> Optional[str]:
    for p in MUSE_REPO_CANDIDATES:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "scripts", "inference.py")):
            return p
    # fallback scan in volume (limit depth)
    if os.path.isdir(RUNPOD_VOLUME_PATH):
        for root, dirnames, _ in os.walk(RUNPOD_VOLUME_PATH):
            rel = os.path.relpath(root, RUNPOD_VOLUME_PATH)
            if rel.count(os.sep) > 6:
                dirnames[:] = []
                continue
            if os.path.isfile(os.path.join(root, "scripts", "inference.py")) and ("MuseTalk" in root):
                return root
    return None

def _require_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise RuntimeError(f"Missing {label}: {path}")

def _json_load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _json_save(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _find_newest_mp4(search_root: str) -> Optional[str]:
    newest = None
    newest_mtime = 0.0
    for root, _, files in os.walk(search_root):
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
    return newest

# --------------------------------------------------
# Container sanity check
# --------------------------------------------------
def _container_import_check() -> Dict[str, Any]:
    env = _clean_env(None)
    cmd = [PY_CONTAINER, "-c", "import cv2, mmcv, mmengine; print('OK_CONTAINER_IMPORTS')"]
    code, out = _run(cmd, env=env, timeout=SCAN_TIMEOUT_SEC)
    return {"ok": code == 0, "code": code, "out_tail": _tail(out)}

# --------------------------------------------------
# MuseTalk runner (edita config en TEMP, NO toca el repo)
# --------------------------------------------------
def _musetalk_infer(repo_root: str, input_mp4: str, audio_wav: str) -> Dict[str, Any]:
    _require_file(os.path.join(repo_root, "scripts", "inference.py"), "MuseTalk inference.py")

    base_cfg_path = os.path.join(repo_root, "inference_config.json")
    _require_file(base_cfg_path, "inference_config.json")

    # MuseTalk normalmente usa rutas dentro del config; hacemos una copia temporal con tus inputs.
    tmpdir = tempfile.mkdtemp(prefix="musetalk_cfg_")
    try:
        cfg = _json_load(base_cfg_path)

        # 🔧 Ajustes típicos (best-effort). Si tus keys difieren, igual dejamos el cfg original y fallará con log claro.
        # Muchos forks usan keys similares a: "video_path" / "audio_path" / "input_video" / "input_audio"
        # Probamos setear varias sin romper:
        for k in ("video_path", "input_video", "video", "source_video"):
            if k in cfg:
                cfg[k] = input_mp4
        for k in ("audio_path", "input_audio", "audio", "source_audio"):
            if k in cfg:
                cfg[k] = audio_wav

        # Algunos usan nested:
        if isinstance(cfg.get("input"), dict):
            cfg["input"]["video_path"] = cfg["input"].get("video_path", input_mp4)
            cfg["input"]["audio_path"] = cfg["input"].get("audio_path", audio_wav)

        tmp_cfg_path = os.path.join(tmpdir, "inference_config.json")
        _json_save(tmp_cfg_path, cfg)

        env = _clean_env(repo_root)

        cmd = [
            PY_CONTAINER, "-u", "scripts/inference.py",
            "--inference_config", tmp_cfg_path,
        ]

        code, out = _run(cmd, cwd=repo_root, env=env, timeout=HARD_TIMEOUT_SEC)
        if code != 0:
            raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

        # Buscar el mp4 más nuevo dentro del repo (MuseTalk usualmente escribe ahí)
        out_mp4 = _find_newest_mp4(repo_root)

        return {
            "ok": True,
            "python_used": PY_CONTAINER,
            "output_mp4_guess": out_mp4,
            "log_tail": _tail(out),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# --------------------------------------------------
# Modes
# --------------------------------------------------
def mode_scan() -> Dict[str, Any]:
    repo_root = _pick_musetalk_repo()
    chk = _container_import_check()
    return {
        "ok": True,
        "msg": "SCAN_OK",
        "repo_root": repo_root,
        "py_container": PY_CONTAINER,
        "container_imports": chk,
    }

def mode_echo() -> Dict[str, Any]:
    repo_root = _pick_musetalk_repo()
    chk = _container_import_check()

    # extra: imprime versión python
    env = _clean_env(None)
    code, out = _run([PY_CONTAINER, "-V"], env=env, timeout=SCAN_TIMEOUT_SEC)

    return {
        "ok": True,
        "msg": "ECHO_OK",
        "base": RUNPOD_VOLUME_PATH,
        "repo_root": repo_root,
        "py_container": PY_CONTAINER,
        "python_version": _tail(out, 200),
        "container_imports": chk,
    }

def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    start = _now()
    repo_root = _pick_musetalk_repo()
    if not repo_root:
        raise RuntimeError("MuseTalk repo not found on volume (expected /runpod-volume/volume_old/MuseTalk)")

    chk = _container_import_check()
    if not chk.get("ok"):
        raise RuntimeError("Container missing deps (cv2/mmcv/mmengine). Fix Dockerfile.\n" + chk.get("out_tail", ""))

    tmp = tempfile.mkdtemp(prefix="v2v_")
    try:
        in_mp4 = os.path.join(tmp, "input.mp4")
        wav = os.path.join(tmp, "audio.wav")

        # video
        if _looks_like_url(inp.get("video_url")):
            _download(inp["video_url"], in_mp4)
        elif isinstance(inp.get("video_b64"), str) and inp["video_b64"]:
            _b64_to_file(inp["video_b64"], in_mp4)
        else:
            raise RuntimeError("Missing video_url or video_b64")

        # audio
        if _looks_like_url(inp.get("audio_url")):
            _download(inp["audio_url"], wav)
        elif isinstance(inp.get("audio_b64"), str) and inp["audio_b64"]:
            _b64_to_file(inp["audio_b64"], wav)
        else:
            raise RuntimeError("Missing audio_url OR audio_b64 (este worker no genera XTTS)")

        info = _musetalk_infer(repo_root, in_mp4, wav)
        elapsed = int((_now() - start) * 1000)

        return {
            "ok": True,
            "msg": "VOICE2VIDEO_OK",
            "execution_ms": elapsed,
            "repo_root": repo_root,
            "python_used": info.get("python_used"),
            "output_mp4_guess": info.get("output_mp4_guess"),
            "log_tail": info.get("log_tail"),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------
# Handler
# --------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp = event.get("input") if isinstance(event, dict) else None
        if not isinstance(inp, dict):
            return {"ok": False, "error": "Missing or invalid input (expected JSON with field 'input')"}

        mode = str(inp.get("mode", "scan")).strip().lower()

        if mode == "scan":
            return mode_scan()
        if mode == "echo":
            return mode_echo()
        if mode in ("voice2video", "v2v"):
            return mode_voice2video(inp)

        return {"ok": False, "error": f"Unknown mode: {mode}"}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

runpod.serverless.start({"handler": handler})
