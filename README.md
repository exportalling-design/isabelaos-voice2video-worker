# IsabelaOS Voice2Video Worker (XTTS + MuseTalk)

## Env vars
- MUSE_ROOT=/runpod-volume/MuseTalk
- VOICES_DIR=/runpod-volume/voices
- FEMALE_REF_WAV=/runpod-volume/voices/female_ref.wav
- MALE_REF_WAV=/runpod-volume/voices/male_ref.wav
- MUSE_PY=python (o /runpod-volume/.../venv/bin/python si MuseTalk usa venv)

## Input
{
  "mode": "voice_to_video",
  "text": "Hola...",
  "voice": "female" | "male",
  "lang": "es" | "en",
  "seconds": 3 | 5,
  "video_b64": "...",   // o video_url
  "video_url": "https://..."
}
