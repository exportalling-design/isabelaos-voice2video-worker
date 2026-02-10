# /app/xtts_generate.py
import os
import argparse
from TTS.api import TTS

# ✅ Auto-aceptar términos Coqui XTTS (NO interactivo)
os.environ["COQUI_TOS_AGREED"] = "1"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--lang", default="es")
    ap.add_argument("--speaker_wav", default="")
    args = ap.parse_args()

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    kwargs = {
        "text": args.text,
        "file_path": args.out_wav,
        "language": args.lang,
    }

    if args.speaker_wav and os.path.exists(args.speaker_wav):
        kwargs["speaker_wav"] = args.speaker_wav

    tts.tts_to_file(**kwargs)
    print("XTTS_OK", args.out_wav)

if __name__ == "__main__":
    main()
