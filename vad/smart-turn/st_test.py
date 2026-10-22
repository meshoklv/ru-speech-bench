"""Smart Turn v3: first smoke test on one recording (cut-mid-segment vs end-of-speech).

Usage:
    pip install onnxruntime transformers silero-vad torch numpy soundfile
    SMART_TURN_MODEL=./smart-turn-v3.2-cpu.onnx python3 st_test.py recording_16k.wav
Prints model I/O shapes, p(finished) per case and CPU inference latency.
Superseded by st_test2.py / st_ab.py for anything beyond a smoke test.
"""
import time, sys, os, resource
import numpy as np, onnxruntime as ort, soundfile as sf
from transformers import WhisperFeatureExtractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import truncate_audio_to_last_n_seconds

M = os.environ.get("SMART_TURN_MODEL", "./smart-turn-v3.2-cpu.onnx")
so = ort.SessionOptions(); so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.inter_op_num_threads = 1; so.intra_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
t0=time.time(); sess = ort.InferenceSession(M, sess_options=so); print(f"model load: {time.time()-t0:.2f}s")
print("inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
fe = WhisperFeatureExtractor(chunk_length=8)

def predict(audio):
    a = truncate_audio_to_last_n_seconds(audio, 8)
    inp = fe(a, sampling_rate=16000, return_tensors="np", padding="max_length",
             max_length=8*16000, truncation=True, do_normalize=True)
    feats = inp.input_features.astype(np.float32)
    t=time.time(); out = sess.run(None, {sess.get_inputs()[0].name: feats}); el=(time.time()-t)*1000
    p = float(np.ravel(out[0])[0])
    if not (0.0 <= p <= 1.0): p = 1/(1+np.exp(-p))
    return p, el

# ---------- build test cases from real Russian speech ----------
import torch; torch.set_num_threads(4)
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
WAV = sys.argv[1]
vad = load_silero_vad()
wav = read_audio(WAV, sampling_rate=16000)
ts = get_speech_timestamps(wav, vad, sampling_rate=16000, return_seconds=True)
a = wav.numpy()
SR=16000

complete, incomplete = [], []
for i, s in enumerate(ts):
    gap_after = (ts[i+1]['start'] - s['end']) if i+1 < len(ts) else 99
    seg_len = s['end'] - s['start']
    # COMPLETE: speech ends and a long pause (>1.2 s) follows
    if gap_after > 1.2 and seg_len > 1.0:
        st = max(0, s['end']-8.0)
        complete.append((f"end@{s['end']:.1f}s gap={gap_after:.1f}s", a[int(st*SR):int(s['end']*SR)]))
    # INCOMPLETE: cut in the MIDDLE of a long segment
    if seg_len > 3.0:
        cut = s['start'] + seg_len*0.55
        st = max(0, cut-8.0)
        incomplete.append((f"cut@{cut:.1f}s mid-seg", a[int(st*SR):int(cut*SR)]))

complete, incomplete = complete[:15], incomplete[:15]
print(f"\ntest cases: complete={len(complete)}  incomplete={len(incomplete)}")

lat=[]
def run(cases, label, expect):
    ok=0
    for name, aud in cases:
        p, el = predict(aud); lat.append(el)
        pred = 1 if p>0.5 else 0
        ok += (pred==expect)
        print(f"  [{label}] {name:28s} p(finished)={p:.3f} -> {'FINISHED' if pred else 'continues':10s} {'OK' if pred==expect else 'MISS'}")
    print(f"  {label}: {ok}/{len(cases)} = {100*ok/max(1,len(cases)):.0f}%")
    return ok, len(cases)

print("\n=== EXPECT 'turn finished' (speech before a long pause) ===")
o1,n1 = run(complete, "compl", 1)
print("\n=== EXPECT 'speaker continues' (cut mid-phrase) ===")
o2,n2 = run(incomplete, "incom", 0)
print(f"\nTOTAL: {o1+o2}/{n1+n2} = {100*(o1+o2)/max(1,n1+n2):.1f}%")
lat=np.array(lat)
print(f"CPU inference: mean={lat.mean():.1f} ms  median={np.median(lat):.1f} ms  p95={np.percentile(lat,95):.1f} ms  min={lat.min():.1f} max={lat.max():.1f} (n={len(lat)}, 1 thread)")
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
