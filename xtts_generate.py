import os
import sys
from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"

text = sys.argv[1]
out_wav = sys.argv[2]
speaker_wav = sys.argv[3] if len(sys.argv) > 3 else None
lang = sys.argv[4] if len(sys.argv) > 4 else "es"

tts = TTS(
    "tts_models/multilingual/multi-dataset/xtts_v2",
    gpu=True
)

kwargs = dict(
    text=text,
    file_path=out_wav,
    language=lang,
)

if speaker_wav and os.path.exists(speaker_wav):
    kwargs["speaker_wav"] = speaker_wav

tts.tts_to_file(**kwargs)
