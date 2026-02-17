import argparse
import os
import sys

def _sanitize_sys_path():
    # Evita mezclar site-packages del volumen (/runpod-volume) con la imagen
    bad_prefixes = ("/runpod-volume/", "/workspace/")
    sys.path = [p for p in sys.path if not any(p.startswith(b) for b in bad_prefixes)]
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)

def main():
    _sanitize_sys_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="es")
    ap.add_argument("--speaker_wav", required=True)
    ap.add_argument("--out_wav", required=True)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # ✅ Auto-acepta CPML (evita prompt y/n)
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    # Import DESPUÉS de sanear path
    from TTS.api import TTS

    model_name = os.environ.get("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    use_gpu = os.environ.get("TTS_USE_GPU", "1").strip().lower() not in ("0", "false")

    tts = TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)

    tts.tts_to_file(
        text=args.text,
        speaker_wav=args.speaker_wav,
        language=args.lang,
        file_path=args.out_wav
    )

    print(f"[OK] wrote wav: {args.out_wav}")

if __name__ == "__main__":
    main()
