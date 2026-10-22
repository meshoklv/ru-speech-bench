"""Smart Turn v3 CPU latency vs. intra-op thread count (random-noise input).

Usage:
    pip install onnxruntime transformers numpy
    SMART_TURN_MODEL=./smart-turn-v3.2-cpu.onnx python3 st_lat.py
Prints median/p95 inference latency for 1/2/4/8 threads and the cost of the
Whisper feature extractor. Run on an idle machine.
"""
import os, time, numpy as np, onnxruntime as ort
MODEL = os.environ.get("SMART_TURN_MODEL", "./smart-turn-v3.2-cpu.onnx")
from transformers import WhisperFeatureExtractor
fe = WhisperFeatureExtractor(chunk_length=8)
aud = np.random.randn(8*16000).astype(np.float32)*0.01
feats = fe(aud, sampling_rate=16000, return_tensors="np", padding="max_length",
           max_length=8*16000, truncation=True, do_normalize=True).input_features.astype(np.float32)
for th in (1,2,4,8):
    so = ort.SessionOptions(); so.intra_op_num_threads=th; so.inter_op_num_threads=1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(MODEL, sess_options=so)
    n = s.get_inputs()[0].name
    for _ in range(5): s.run(None, {n: feats})          # warm-up
    t=[]
    for _ in range(30):
        t0=time.perf_counter(); s.run(None, {n: feats}); t.append((time.perf_counter()-t0)*1000)
    t=np.array(t); print(f"intra_op_threads={th}: median={np.median(t):6.1f} ms  p95={np.percentile(t,95):6.1f} ms  min={t.min():6.1f}")
# cost of the mel-spectrogram features alone
t=[]
for _ in range(20):
    t0=time.perf_counter()
    fe(aud, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=8*16000, truncation=True, do_normalize=True)
    t.append((time.perf_counter()-t0)*1000)
print(f"WhisperFeatureExtractor (mel features): median={np.median(t):.1f} ms")
