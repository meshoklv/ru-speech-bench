"""Silero VAD: segmentation of a recording + per-chunk latency.

Usage:
    pip install silero-vad torch
    python3 silero_test.py AUDIO.wav 16000     # second argument: 16000 or 8000
Prints segment count, speech/silence share, whole-file processing time and the
latency of one 32 ms chunk (512 samples at 16 kHz, 256 at 8 kHz).
"""
import sys, time, os, resource
import numpy as np, torch
torch.set_num_threads(4)
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

wav_path, sr = sys.argv[1], int(sys.argv[2])
t0=time.time(); model = load_silero_vad(onnx=False); t_load=time.time()-t0
wav = read_audio(wav_path, sampling_rate=sr)
dur = len(wav)/sr
t0=time.time()
ts = get_speech_timestamps(wav, model, sampling_rate=sr, return_seconds=True)
t_full=time.time()-t0
print(f"file={os.path.basename(wav_path)} sr={sr} dur={dur:.1f}s load={t_load:.2f}s")
print(f"segments={len(ts)} vad_time={t_full:.2f}s RTF={t_full/dur:.4f} speedup=x{dur/t_full:.0f}")
speech = sum(s['end']-s['start'] for s in ts)
print(f"speech={speech:.1f}s ({100*speech/dur:.0f}%)  silence={dur-speech:.1f}s")
print("first 6 segments:", [(round(s['start'],2), round(s['end'],2)) for s in ts[:6]])
# longest contiguous
lg = max(ts, key=lambda s: s['end']-s['start'])
print(f"LONGEST: {lg['start']:.2f}-{lg['end']:.2f} ({lg['end']-lg['start']:.2f}s)")
# per-chunk latency (30ms @16k = 512 samples; @8k = 256)
n = 512 if sr==16000 else 256
model.reset_states()
chunks = [wav[i:i+n] for i in range(0, len(wav)-n, n)][:2000]
t0=time.time()
for c in chunks: model(c, sr)
el=(time.time()-t0)/len(chunks)*1000
print(f"per-chunk ({n} samples = {1000*n/sr:.0f}ms): {el:.3f} ms  -> real-time factor x{(1000*n/sr)/el:.0f}")
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
