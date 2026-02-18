import os
import argparse
import traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="es")
    parser.add_argument("--speaker_wav", required=True)
    parser.add_argument("--out_wav", required=True)
    parser.add_argument("--model_name", default="tts_models/multilingual/multi-dataset/xtts_v2")
    args = parser.parse_args()

    # Coqui ToS
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    from TTS.api import TTS
    import torch

    use_gpu_env = os.environ.get("TTS_USE_GPU", "1").strip().lower() not in ("0", "false", "no")
    device = "cuda" if (use_gpu_env and torch.cuda.is_available()) else "cpu"

    def synth(dev: str):
        tts = TTS(model_name=args.model_name, progress_bar=False)
        try:
            tts.to(dev)
        except Exception:
            # si algo raro pasa, cae a cpu
            dev = "cpu"
            tts.to("cpu")
        tts.tts_to_file(
            text=args.text,
            speaker_wav=args.speaker_wav,
            language=args.lang,
            file_path=args.out_wav
        )
        return dev

    try:
        used = synth(device)
        print(f"OK: wrote {args.out_wav} (device={used})")
        return
    except RuntimeError as e:
        # 🔥 fallback CPU ante ECC / CUDA issues
        msg = str(e).lower()
        if "ecc" in msg or "cuda error" in msg or "device-side assert" in msg:
            try:
                used = synth("cpu")
                print(f"OK: wrote {args.out_wav} (device={used})")
                return
            except Exception:
                print("FATAL after CPU fallback:")
                traceback.print_exc()
                raise
        raise
    except Exception:
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
