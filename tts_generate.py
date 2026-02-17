# /app/tts_generate.py
import argparse
import os
import sys
import traceback

def _sanitize_sys_path():
    # Quita cualquier site-packages del volumen para evitar mezclar cosas raras
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
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    from TTS.api import TTS

    model_name = os.environ.get("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
    want_gpu = os.environ.get("TTS_USE_GPU", "1").strip() not in ("0", "false", "False")

    # 1) Intento GPU
    if want_gpu:
        try:
            tts = TTS(model_name=model_name, progress_bar=False, gpu=True)
            tts.tts_to_file(
                text=args.text,
                speaker_wav=args.speaker_wav,
                language=args.lang,
                file_path=args.out_wav
            )
            print(f"[OK][GPU] wrote wav: {args.out_wav}")
            return
        except Exception as e:
            # Si es ECC u otro error CUDA, cae a CPU
            msg = str(e)
            print("[WARN] GPU TTS failed, fallback to CPU. Err:", msg)
            # no re-raise

    # 2) CPU fallback
    tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
    tts.tts_to_file(
        text=args.text,
        speaker_wav=args.speaker_wav,
        language=args.lang,
        file_path=args.out_wav
    )
    print(f"[OK][CPU] wrote wav: {args.out_wav}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[FATAL] tts_generate failed")
        print(traceback.format_exc())
        raise
