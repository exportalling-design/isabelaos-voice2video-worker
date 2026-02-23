# /app/worker.py
# RunPod Serverless Worker — IsabelaOS Voice2Video (XTTS -> MuseTalk)
# ✅ No reinstala nada
# ✅ Encuentra el python correcto (venv) donde YA existe: cv2 + mmcv + mmengine + mmpose
# ✅ Busca en /runpod-volume y /workspace
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

# --------------------------------------------------
# ENV
# --------------------------------------------------
RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
COQUI_TOS_AGREED = "1"
HARD_TIMEOUT_SEC = int(os.environ.get("HARD_TIMEOUT_SEC", "560"))   # < 600
SCAN_TIMEOUT_SEC = int(os.environ.get("SCAN_TIMEOUT_SEC", "20"))    # per candidate

# Paths on volume
VOICES_DIR = os.path.join(RUNPOD_VOLUME_PATH, "voices")
FEMALE_REF_WAV = os.path.join(VOICES_DIR, "female_ref.wav")
MALE_REF_WAV = os.path.join(VOICES_DIR, "male_ref.wav")

# MuseTalk repo candidates
MUSE_REPO_CANDIDATES = [
    os.path.join(RUNPOD_VOLUME_PATH, "volume_old", "MuseTalk"),
    os.path.join(RUNPOD_VOLUME_PATH, "MuseTalk"),
    os.path.join("/workspace", "volume_old", "MuseTalk"),
    os.path.join("/workspace", "MuseTalk"),
]

# Cache selection
_SELECTED_PY: Optional[str] = None
_SELECTED_REPORT: Optional[Dict[str, Any]] = None


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _now() -> float:
    return time.time()

def _tail(s: str, n: int = 1400) -> str:
    return (s or "")[-n:]

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

def _clean_env(repo_root: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["COQUI_TOS_AGREED"] = "1"
    if repo_root:
        env["PYTHONPATH"] = repo_root + (
            ":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
        )
    return env

def _looks_like_url(s: Any) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst_path, "wb") as f:
        shutil.copyfileobj(r, f)


# --------------------------------------------------
# MuseTalk repo detection
# --------------------------------------------------
def _pick_musetalk_repo() -> Optional[str]:
    for p in MUSE_REPO_CANDIDATES:
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "scripts", "inference.py")):
            return p
    # fallback scan in volume (limit depth)
    for root, dirnames, filenames in os.walk(RUNPOD_VOLUME_PATH):
        rel = os.path.relpath(root, RUNPOD_VOLUME_PATH)
        if rel.count(os.sep) > 6:
            dirnames[:] = []
            continue
        if os.path.isfile(os.path.join(root, "scripts", "inference.py")) and "MuseTalk" in root:
            return root
    return None


# --------------------------------------------------
# Python/Venv auto-detect
# --------------------------------------------------
def _preferred_python_paths() -> List[str]:
    # probamos python, python3, python3.11 porque tus venvs a veces solo linkean a python3
    venv_roots = [
        os.path.join(RUNPOD_VOLUME_PATH, "musetalk_ok"),
        os.path.join(RUNPOD_VOLUME_PATH, "musetalk_ok_persist"),
        os.path.join("/workspace", "musetalk_ok"),
        os.path.join("/workspace", "musetalk_ok_persist"),
    ]
    exes = []
    for vr in venv_roots:
        for name in ("python", "python3", "python3.11"):
            exes.append(os.path.join(vr, "bin", name))
    return exes

def _list_candidate_pythons() -> List[str]:
    candidates = []
    seen = set()

    # 1) preferidos
    for p in _preferred_python_paths():
        if os.path.isfile(p) and os.access(p, os.X_OK) and p not in seen:
            candidates.append(p)
            seen.add(p)

    # 2) buscar bin/python* en /runpod-volume y /workspace (limit depth)
    for base in (RUNPOD_VOLUME_PATH, "/workspace"):
        if not os.path.isdir(base):
            continue
        for root, dirnames, filenames in os.walk(base):
            rel = os.path.relpath(root, base)
            if rel.count(os.sep) > 6:
                dirnames[:] = []
                continue
            if root.endswith("/bin"):
                for name in ("python", "python3", "python3.11"):
                    p = os.path.join(root, name)
                    if os.path.isfile(p) and os.access(p, os.X_OK) and p not in seen:
                        candidates.append(p)
                        seen.add(p)

    # 3) fallback container python
    sys_py = shutil.which("python3") or "/usr/local/bin/python3" or "python3"
    if sys_py not in seen:
        candidates.append(sys_py)

    return candidates

def _probe_python(py: str) -> Dict[str, Any]:
    # probamos por partes para score
    env = _clean_env(None)

    c1, o1 = _run([py, "-c", "import cv2; print('OK_cv2')"], env=env, timeout=SCAN_TIMEOUT_SEC)
    ok_cv2 = (c1 == 0)

    c2, o2 = _run([py, "-c", "import mmcv; import mmengine; print('OK_mmcv')"], env=env, timeout=SCAN_TIMEOUT_SEC)
    ok_mmcv = (c2 == 0)

    c3, o3 = _run([py, "-c", "import mmpose; print('OK_mmpose')"], env=env, timeout=SCAN_TIMEOUT_SEC)
    ok_mmpose = (c3 == 0)

    # score: lo que necesitamos para tu caso
    score = 0
    if ok_cv2:
        score += 1
    if ok_mmcv:
        score += 3
    if ok_mmpose:
        score += 2

    return {
        "py": py,
        "score": score,
        "ok_cv2": ok_cv2,
        "ok_mmcv_mmengine": ok_mmcv,
        "ok_mmpose": ok_mmpose,
        "tail_cv2": _tail(o1),
        "tail_mmcv": _tail(o2),
        "tail_mmpose": _tail(o3),
    }

def _select_best_python() -> Tuple[str, Dict[str, Any]]:
    global _SELECTED_PY, _SELECTED_REPORT
    if _SELECTED_PY and _SELECTED_REPORT:
        return _SELECTED_PY, _SELECTED_REPORT

    cands = _list_candidate_pythons()
    results = []
    for py in cands:
        results.append(_probe_python(py))

    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    best = results_sorted[0]
    _SELECTED_PY = best["py"]
    _SELECTED_REPORT = {"best": best, "top": results_sorted[:12], "candidates_count": len(cands)}
    return _SELECTED_PY, _SELECTED_REPORT


# --------------------------------------------------
# MuseTalk runner
# --------------------------------------------------
def _musetalk_infer(repo_root: str, input_mp4: str, audio_wav: str) -> Dict[str, Any]:
    py, report = _select_best_python()

    # aseguramos PYTHONPATH hacia repo (MuseTalk como repo)
    env = _clean_env(repo_root)

    # NOTA: MuseTalk suele leer input/audio desde inference_config.json
    # (Aquí no editamos config para no tocar tu setup actual)
    cmd = [
        py, "-u", "scripts/inference.py",
        "--inference_config", "inference_config.json",
    ]

    code, out = _run(cmd, cwd=repo_root, env=env, timeout=HARD_TIMEOUT_SEC)
    if code != 0:
        raise RuntimeError("MuseTalk inference failed\n" + _tail(out))

    # buscar mp4 más nuevo (best-effort)
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
        "python_best": report.get("best"),
        "python_top": report.get("top"),
        "output_mp4_guess": newest,
        "log_tail": _tail(out),
    }


# --------------------------------------------------
# Modes
# --------------------------------------------------
def mode_scan() -> Dict[str, Any]:
    repo_root = _pick_musetalk_repo()
    py, report = _select_best_python()
    return {
        "ok": True,
        "msg": "SCAN_OK",
        "repo_root": repo_root,
        "picked_python": py,
        "best": report.get("best"),
        "top": report.get("top"),
        "candidates_count": report.get("candidates_count"),
    }

def mode_echo() -> Dict[str, Any]:
    repo_root = _pick_musetalk_repo()
    py, report = _select_best_python()

    return {
        "ok": True,
        "msg": "ECHO_OK",
        "base": RUNPOD_VOLUME_PATH,
        "repo_root": repo_root,
        "picked_python": py,
        "best": report.get("best"),
        "top": report.get("top"),
    }

def mode_voice2video(inp: Dict[str, Any]) -> Dict[str, Any]:
    start = _now()
    repo_root = _pick_musetalk_repo()
    if not repo_root:
        raise RuntimeError("MuseTalk repo not found on volume")

    tmp = tempfile.mkdtemp(prefix="v2v_")
    try:
        in_mp4 = os.path.join(tmp, "input.mp4")
        wav = os.path.join(tmp, "audio.wav")

        if _looks_like_url(inp.get("video_url")):
            _download(inp["video_url"], in_mp4)
        elif isinstance(inp.get("video_b64"), str) and inp["video_b64"]:
            with open(in_mp4, "wb") as f:
                f.write(base64.b64decode(inp["video_b64"]))
        else:
            raise RuntimeError("Missing video_url or video_b64")

        if _looks_like_url(inp.get("audio_url")):
            _download(inp["audio_url"], wav)
        elif isinstance(inp.get("audio_b64"), str) and inp["audio_b64"]:
            with open(wav, "wb") as f:
                f.write(base64.b64decode(inp["audio_b64"]))
        else:
            raise RuntimeError("Missing audio_url OR audio_b64 (en este worker no generamos XTTS)")

        info = _musetalk_infer(repo_root, in_mp4, wav)
        elapsed = int((_now() - start) * 1000)

        return {
            "ok": True,
            "msg": "VOICE2VIDEO_OK",
            "execution_ms": elapsed,
            "repo_root": repo_root,
            "python_used": info.get("python_used"),
            "python_best": info.get("python_best"),
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
