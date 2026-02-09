# /app/tts_generate.py
import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="es")
    ap.add_argument("--speaker_wav", required=True)
    ap.add_argument("--out_wav", required=True)
    args = ap.parse_args()

    # Evitar warnings raros y forzar cosas estables
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Import aquí para que si falla, el error salga claro
    from TTS.api import TTS

    # Modelo XTTS v2 (muy usado)
    # Si querés fijarlo a un path local, lo cambiamos luego.
    model_name = os.environ.get("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

    # GPU si está disponible
    use_gpu = os.environ.get("TTS_USE_GPU", "1").strip() not in ("0", "false", "False")

    tts = TTS(model_name=model_name, progress_bar=False, gpu=use_gpu)

    # Genera wav
    tts.tts_to_file(
        text=args.text,
        speaker_wav=args.speaker_wav,
        language=args.lang,
        file_path=args.out_wav
    )

    print(f"[OK] wrote wav: {args.out_wav}")

if __name__ == "__main__":
    main()
