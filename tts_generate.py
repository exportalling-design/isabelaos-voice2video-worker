#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError("CMD_FAILED: " + " ".join(cmd) + "\n" + (p.stdout or ""))
    return p.stdout or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--lang", default="es")
    ap.add_argument("--voice", default="female", choices=["female","male"])
    ap.add_argument("--out_wav", required=True)
    args = ap.parse_args()

    # Model paths (ponelos en /runpod-volume para que persistan)
    # Puedes cambiar estos envs en RunPod:
    #  - PIPER_FEMALE_MODEL
    #  - PIPER_MALE_MODEL
    female_model = os.environ.get("PIPER_FEMALE_MODEL", "/runpod-volume/voices/piper/female.onnx")
    male_model   = os.environ.get("PIPER_MALE_MODEL",   "/runpod-volume/voices/piper/male.onnx")

    model = female_model if args.voice == "female" else male_model
    if not os.path.exists(model):
        raise RuntimeError(f"Piper model not found: {model}")

    # Piper: texto -> wav
    # Nota: Piper ignora --lang; el “latino” depende del modelo que descargues (es_MX, es_419, etc).
    cmd = ["piper", "--model", model, "--output_file", args.out_wav]
    # Pasamos el texto por stdin
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate(args.text)
    if p.returncode != 0:
        raise RuntimeError("CMD_FAILED: " + " ".join(cmd) + "\n" + (out or ""))

if __name__ == "__main__":
    main()
