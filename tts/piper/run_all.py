"""Piper TTS benchmark: synthesis speed for the four Russian `medium` voices.

For every voice x phrase it synthesizes a WAV, and records load time, synthesis
time, audio duration and "speed" = audio_seconds / synthesis_seconds (higher is
faster; > 1 means faster than real time; this is the inverse of the usual RTF).

Usage:
    pip install piper-tts
    # put ru_RU-<voice>-medium.onnx(+.json) under $PIPER_VOICES_DIR/<voice>/
    PIPER_VOICES_DIR=./voices OUT_DIR=./out python3 run_all.py
Outputs: $OUT_DIR/<voice>-<phrase>.wav and $OUT_DIR/bench_results.json
"""
import json
import os
import resource
import time
import wave

from piper import PiperVoice

from phrases import PHRASES

VOICES_DIR = os.environ.get("PIPER_VOICES_DIR", "./voices")
OUT_DIR = os.environ.get("OUT_DIR", "./out")
VOICE_NAMES = ["denis", "dmitri", "irina", "ruslan"]

os.makedirs(OUT_DIR, exist_ok=True)

results = []
for vname in VOICE_NAMES:
    model_path = os.path.join(VOICES_DIR, vname, f"ru_RU-{vname}-medium.onnx")
    t0 = time.time()
    voice = PiperVoice.load(model_path)
    load_s = time.time() - t0
    for pname, text in PHRASES.items():
        out_wav = os.path.join(OUT_DIR, f"{vname}-{pname}.wav")
        t0 = time.time()
        with wave.open(out_wav, "wb") as wf:
            voice.synthesize_wav(text, wf)
        synth_s = time.time() - t0
        with wave.open(out_wav, "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        results.append({
            "voice": vname, "phrase": pname,
            "load_s": round(load_s, 3), "synth_s": round(synth_s, 3),
            "dur_s": round(dur, 3),
            "audio_per_synth": round(dur / synth_s, 2) if synth_s > 0 else None,
        })
        print(f"{vname:8s} {pname:10s} synth={synth_s:.3f}s dur={dur:.3f}s "
              f"speed={dur / synth_s:.2f}x")

peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"\npeak_rss_mb={peak:.1f}")
with open(os.path.join(OUT_DIR, "bench_results.json"), "w") as f:
    json.dump({"results": results, "peak_rss_mb": peak}, f, ensure_ascii=False, indent=2)
