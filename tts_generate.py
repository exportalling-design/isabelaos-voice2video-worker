import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="es")
    ap.add_argument("--speaker_wav", required=True)
    ap.add_argument("--out_wav", required=True)
    args = ap.parse_args()

    if not os.path.isfile(args.speaker_wav):
        raise SystemExit(f"Missing speaker_wav: {args.speaker_wav}")

    # Import dentro del env XTTS
    from TTS.api import TTS
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tts.tts_to_file(
        text=args.text,
        speaker_wav=args.speaker_wav,
        language=args.lang,
        file_path=args.out_wav,
    )

    print(f"OK -> {args.out_wav}")

if __name__ == "__main__":
    main()
