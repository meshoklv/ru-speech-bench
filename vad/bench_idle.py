"""Silero VAD per-chunk latency: torch vs ONNX, 16 kHz vs 8 kHz.

Usage:
    pip install silero-vad torch
    WAV_16K=./audio/full16k.wav WAV_8K=./audio/full8k.wav python3 bench_idle.py
Run it on an otherwise idle machine; it is single-threaded by design.
"""
import os, time, numpy as np, torch, resource

# one recording resampled to 16 kHz and 8 kHz; override the paths via env vars
WAV = {16000: os.environ.get("WAV_16K", "./audio/full16k.wav"),
       8000: os.environ.get("WAV_8K", "./audio/full8k.wav")}
torch.set_num_threads(1)
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
print("=== SILERO VAD (idle machine, 1 thread) ===")
for onnx in (False, True):
    m = load_silero_vad(onnx=onnx)
    for sr in (16000, 8000):
        n = 512 if sr==16000 else 256
        wav = read_audio(WAV[sr], sampling_rate=sr)
        chunks=[wav[i:i+n] for i in range(0,len(wav)-n,n)][:3000]
        m.reset_states()
        for c in chunks[:50]: m(c, sr)           # warm-up
        t=[]
        for c in chunks:
            t0=time.perf_counter(); m(c, sr); t.append((time.perf_counter()-t0)*1e3)
        t=np.array(t)
        print(f"  {'ONNX ' if onnx else 'torch'} {sr//1000}kHz chunk {1000*n/sr:.0f}ms: median={np.median(t):.3f} ms p95={np.percentile(t,95):.3f} ms -> {(1000*n/sr)/np.median(t):.0f}x faster than real time")
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
