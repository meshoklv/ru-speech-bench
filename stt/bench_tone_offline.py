"""Offline T-one benchmark (greedy decoder): 16 kHz vs telephone-style 8 kHz clips, RTF, RAM.

Requires `pip install -e T-one` (voicekit-team/T-one), onnxruntime, huggingface_hub.
Test clips (<key>-16k.wav / <key>-phone8k.wav) are read from $TESTAUDIO_DIR (default ./testaudio);
they are not shipped with this repo - supply your own recordings.
Writes results_offline.json next to this script.
"""
import sys, time, json, resource, os
from pathlib import Path
import numpy as np
from tone import StreamingCTCPipeline, read_audio
from tone.decoder import DecoderType

AUDIO_DIR = Path(os.environ.get("TESTAUDIO_DIR", "testaudio"))
KEYS = ["a53", "b79", "c193", "d285"]

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

t0 = time.perf_counter()
pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)
load_s = time.perf_counter() - t0
print(f"### MODEL LOAD: {load_s:.2f} s, peak RSS after load: {rss_mb():.0f} MB", flush=True)

results = {}
for key in KEYS:
    for variant in ["16k", "phone8k"]:
        path = AUDIO_DIR / f"{key}-{variant}.wav"
        audio = read_audio(path)
        dur = len(audio) / 8000
        t = time.perf_counter()
        phrases = pipeline.forward_offline(audio)
        el = time.perf_counter() - t
        text = " ".join(p.text for p in phrases)
        results[f"{key}-{variant}"] = {
            "audio_s": round(dur, 2), "proc_s": round(el, 2),
            "rtf": round(el / dur, 4), "speedup": round(dur / el, 1),
            "n_phrases": len(phrases), "chars": len(text), "text_chars": len(text),
        }
        print(f"[{key}-{variant}] audio {dur:.1f}s | wall {el:.2f}s | "
              f"RTF {el/dur:.4f} | x{dur/el:.1f} real time | phrases {len(phrases)} | chars {len(text)} | peakRSS {rss_mb():.0f} MB", flush=True)

print(f"### PEAK PROCESS MEMORY: {rss_mb():.0f} MB")
out = Path(__file__).parent / "results_offline.json"
out.write_text(json.dumps({"load_s": round(load_s,2), "peak_rss_mb": round(rss_mb()), "runs": results}, ensure_ascii=False, indent=2))
print("saved", out)
