"""Run 1..N parallel T-one streaming lines in one process (one model per process).

Usage: python bench_hostload.py [N]      (needs `pip install -e T-one`, onnxruntime,
huggingface_hub; test clips in $TESTAUDIO_DIR, default ./testaudio, file a53-phone8k.wav)
Prints peak RSS, per-line RTF and whether each line keeps real time.
"""
import os, sys, time, json, threading, resource
from pathlib import Path
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download
from tone.onnx_wrapper import StreamingCTCModel
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.decoder import GreedyCTCDecoder
from tone.pipeline import StreamingCTCPipeline
from tone import read_audio

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
t0 = time.perf_counter()
sess = ort.InferenceSession(hf_hub_download("t-tech/T-one", "model.onnx"), sess_options=so)
dec = GreedyCTCDecoder()
model = StreamingCTCModel(sess)
load_s = time.perf_counter() - t0
print(f"model load: {load_s:.2f}s, RSS {rss():.0f} MB", flush=True)

CHUNK, PAD = StreamingCTCPipeline.CHUNK_SIZE, StreamingCTCPipeline.PADDING
AUDIO = Path(os.environ.get("TESTAUDIO_DIR", "testaudio"))
audio = read_audio(AUDIO / "a53-phone8k.wav")
DUR = len(audio) / 8000
audio = np.pad(audio, (PAD, PAD))

def line(res, idx):
    pipe = StreamingCTCPipeline(model, StreamingLogprobSplitter(), dec)
    state = None
    t0 = time.perf_counter()
    for i in range(0, len(audio), CHUNK):
        c = audio[i:i+CHUNK]; c = np.pad(c, (0, -len(c) % CHUNK))
        _, state = pipe.forward(c, state)
    res[idx] = time.perf_counter() - t0

res = {}
ths = [threading.Thread(target=line, args=(res, i)) for i in range(N)]
t0 = time.perf_counter()
for t in ths: t.start()
for t in ths: t.join()
wall = time.perf_counter() - t0
slow = max(res.values())
cpu_s = resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime
print(f"N={N} lines | audio {DUR:.1f}s | wall {wall:.2f}s | slowest line {slow:.2f}s | "
      f"RTF per line {slow/DUR:.3f} | {'KEEPS' if slow<DUR else 'MISSES'} real time | "
      f"process CPU time {cpu_s:.2f}s ({100*cpu_s/wall:.0f}% of one core) | RSS {rss():.0f} MB", flush=True)
