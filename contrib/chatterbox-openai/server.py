#!/usr/bin/env python3
"""
Chatterbox Turbo as an OpenAI-compatible TTS server.

Exposes exactly the surface pin-clientd's TTS path speaks:

    GET  /v1/models         -> {"data":[{"id":"chatterbox-turbo"}]}
    POST /v1/audio/speech   -> audio bytes (wav or mp3)
      body: {"model":"chatterbox-turbo","input":"...","voice":"mark",
             "response_format":"wav","speed":1.0}

Voices are reference-audio clips in --voices-dir: a request for voice "mark"
uses {voices_dir}/mark.wav as the Chatterbox audio prompt. "default" (or a
missing file) uses the model's built-in voice. Keep voice references
authorized — do not clone anyone without documented permission.

Run on the GPU box:

    pip install chatterbox-tts fastapi uvicorn
    python3 server.py --device cuda --port 8880 --voices-dir ./voices

Smoke test:

    curl -s http://127.0.0.1:8880/v1/audio/speech \
      -H 'content-type: application/json' \
      -d '{"model":"chatterbox-turbo","input":"Welcome to the room.","response_format":"wav"}' \
      -o hello.wav && file hello.wav

Then point pin-clientd at it (config.json, per node):

    "ttsUri": "http://127.0.0.1:8880",
    "ttsModels": ["chatterbox-turbo"]
"""

import argparse
import io
import os
import time

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="chatterbox-openai")
MODEL = None
VOICES_DIR = "./voices"
MODEL_ID = "chatterbox-turbo"


class SpeechRequest(BaseModel):
    model: str = MODEL_ID
    input: str
    voice: str = "default"
    response_format: str = "mp3"
    speed: float | None = None


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "resemble-ai"}]}


@app.get("/health")
def health():
    return {"ok": MODEL is not None, "model": MODEL_ID, "device": str(getattr(MODEL, "device", "unloaded"))}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if MODEL is None:
        raise HTTPException(503, "model not loaded")
    if not req.input.strip():
        raise HTTPException(400, "input is empty")
    if req.response_format not in ("wav", "mp3"):
        raise HTTPException(400, f"unsupported response_format: {req.response_format}")

    kwargs = {}
    ref = os.path.join(VOICES_DIR, f"{req.voice}.wav")
    if req.voice != "default" and os.path.isfile(ref):
        kwargs["audio_prompt_path"] = ref

    t0 = time.time()
    with torch.inference_mode():
        wav = MODEL.generate(req.input, **kwargs)

    buf = io.BytesIO()
    fmt = req.response_format
    torchaudio.save(buf, wav.cpu(), MODEL.sr, format=fmt)
    audio = buf.getvalue()
    print(f"[tts] {len(req.input)} chars -> {len(audio)} bytes ({fmt}) "
          f"voice={req.voice} in {time.time() - t0:.2f}s")
    return Response(
        content=audio,
        media_type="audio/wav" if fmt == "wav" else "audio/mpeg",
        headers={"x-render-seconds": f"{time.time() - t0:.2f}"},
    )


def main():
    global MODEL, VOICES_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8880)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--voices-dir", default="./voices")
    args = ap.parse_args()
    VOICES_DIR = args.voices_dir

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print(f"[boot] loading Chatterbox Turbo on {args.device} ...")
    t0 = time.time()
    MODEL = ChatterboxTurboTTS.from_pretrained(device=args.device)
    print(f"[boot] loaded in {time.time() - t0:.1f}s, sample rate {MODEL.sr}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
