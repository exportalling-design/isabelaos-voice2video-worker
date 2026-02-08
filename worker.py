import os
import json
import subprocess
import tempfile
import runpod

def _exists(p: str) -> bool:
    try:
        return bool(p) and os.path.exists(p)
    except:
        return False

def handler(job):
    inp = job.get("input", {}) or {}
    action = inp.get("action", "healthcheck")

    # --- ENV paths ---
    RUNPOD_VOLUME_PATH = os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume")
    TTS_BIN = os.environ.get("TTS_BIN", f"{RUNPOD_VOLUME_PATH}/xtts_env/bin/tts")
    FEMALE_REF_WAV = os.environ.get("FEMALE_REF_WAV", f"{RUNPOD_VOLUME_PATH}/voices/female_ref.wav")
    MALE_REF_WAV = os.environ.get("MALE_REF_WAV", f"{RUNPOD_VOLUME_PATH}/voices/male_ref.wav")

    # --- Cache dirs (models/cache on volume; temp outputs on /tmp) ---
    os.environ["HOME"] = "/tmp"
    os.environ["TMPDIR"] = "/tmp"
    os.environ["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME", f"{RUNPOD_VOLUME_PATH}/.cache")
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", f"{RUNPOD_VOLUME_PATH}/hf")
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", f"{RUNPOD_VOLUME_PATH}/hf")
    os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", f"{RUNPOD_VOLUME_PATH}/torch")
    os.environ["NUMBA_CACHE_DIR"] = os.environ.get("NUMBA_CACHE_DIR", f"{RUNPOD_VOLUME_PATH}/numba")

    if action == "healthcheck":
        # responde ultra rápido
        tmp_ok = False
        try:
            with tempfile.NamedTemporaryFile(delete=True) as f:
                f.write(b"ok")
            tmp_ok = True
        except:
            tmp_ok = False

        return {
            "ok": True,
            "paths": {
                "RUNPOD_VOLUME_PATH": RUNPOD_VOLUME_PATH,
                "TTS_BIN": TTS_BIN,
                "TTS_BIN_exists": _exists(TTS_BIN),
                "FEMALE_REF_WAV_exists": _exists(FEMALE_REF_WAV),
                "MALE_REF_WAV_exists": _exists(MALE_REF_WAV),
            },
            "tmp_writable": tmp_ok,
        }

    if action == "tts_test":
        text = (inp.get("text") or "Hola, soy IsabelaOS").strip()
        voice = (inp.get("voice") or "female").strip().lower()
        lang = (inp.get("language") or "es").strip()

        speaker = FEMALE_REF_WAV if voice != "male" else MALE_REF_WAV
        out_wav = "/tmp/tts_out.wav"

        cmd = [
            TTS_BIN,
            "--text", text,
            "--model_name", "tts_models/multilingual/multi-dataset/xtts_v2",
            "--speaker_wav", speaker,
            "--language_idx", lang,
            "--out_path", out_wav,
        ]

        # timeout duro para que nunca se quede colgado
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=55)

        if p.returncode != 0:
            return {
                "ok": False,
                "error": "tts_failed",
                "stderr": p.stderr[-2000:],
                "stdout": p.stdout[-2000:],
                "cmd": " ".join(cmd),
            }

        size = os.path.getsize(out_wav) if os.path.exists(out_wav) else 0
        return {
            "ok": True,
            "out_wav": out_wav,
            "bytes": size,
            "stdout_tail": p.stdout[-1200:],
        }

    return {"ok": False, "error": "unknown_action", "action": action}

runpod.serverless.start({"handler": handler})
