"""Smart Turn v3: "speaker finished" vs "pause, then continues" on real speech.

Both groups end at a VAD boundary (as in a real pipeline):
  COMPLETE   = pause after the speech > 3.0 s (the speaker really finished)
  INCOMPLETE = pause after the speech 0.25-0.8 s (breath / hesitation, speech goes on)
NOTE: on monologue recordings (voice messages) this labelling is a weak proxy for
real turn-taking; see docs/engines/smart-turn-v3.md before quoting the numbers.

Usage:
    pip install onnxruntime transformers silero-vad torch numpy
    SMART_TURN_MODEL=./smart-turn-v3.2-cpu.onnx TAIL_MS=200 python3 st_test2.py rec1.wav ...
"""
import time, sys, os, resource
import numpy as np, onnxruntime as ort, torch
MODEL = os.environ.get("SMART_TURN_MODEL", "./smart-turn-v3.2-cpu.onnx")
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import WhisperFeatureExtractor
from audio_utils import truncate_audio_to_last_n_seconds
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

TAIL = float(os.environ.get("TAIL_MS", "200"))/1000.0   # how much trailing silence to keep after the speech
so = ort.SessionOptions(); so.intra_op_num_threads=4; so.inter_op_num_threads=1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(MODEL, sess_options=so)
IN = sess.get_inputs()[0].name
fe = WhisperFeatureExtractor(chunk_length=8)
def predict(a):
    a = truncate_audio_to_last_n_seconds(a, 8)
    f = fe(a, sampling_rate=16000, return_tensors="np", padding="max_length",
           max_length=8*16000, truncation=True, do_normalize=True).input_features.astype(np.float32)
    t0=time.perf_counter(); o = sess.run(None, {IN: f})[0]; el=(time.perf_counter()-t0)*1000
    x = float(np.ravel(o)[0]); p = x if 0<=x<=1 else 1/(1+np.exp(-x))
    return p, el

SR=16000
res_all={}
for WAV in sys.argv[1:]:
    vad = load_silero_vad()
    wav = read_audio(WAV, sampling_rate=16000); a = wav.numpy()
    ts = get_speech_timestamps(wav, vad, sampling_rate=16000, return_seconds=True)
    comp, incomp = [], []
    for i, s in enumerate(ts[:-1]):
        gap = ts[i+1]['start'] - s['end']
        if s['end'] - s['start'] < 0.8: continue
        end = s['end'] + min(TAIL, gap*0.8)
        seg = a[max(0,int((end-8)*SR)):int(end*SR)]
        if gap > 3.0:            comp.append((f"{s['end']:.1f}s pause {gap:.1f}s", seg))
        elif 0.25 <= gap <= 0.8: incomp.append((f"{s['end']:.1f}s pause {gap:.2f}s", seg))
    print(f"\n########## {os.path.basename(WAV)}  (trailing silence {TAIL*1000:.0f} ms)")
    print(f"cases: finished={len(comp)}  pause-then-continue={len(incomp)}")
    lat=[]; tp=fn=tn=fp=0; probs_c=[]; probs_i=[]
    for name,seg in comp:
        p,el=predict(seg); lat.append(el); probs_c.append(p)
        if p>0.5: tp+=1
        else: fn+=1
    for name,seg in incomp:
        p,el=predict(seg); lat.append(el); probs_i.append(p)
        if p>0.5: fp+=1
        else: tn+=1
    n=tp+fn+tn+fp
    if n:
        print(f"  FINISHED        : recognized {tp}/{tp+fn} = {100*tp/max(1,tp+fn):.0f}%   p(median)={np.median(probs_c) if probs_c else -1:.3f}")
        print(f"  CONTINUES       : recognized {tn}/{tn+fp} = {100*tn/max(1,tn+fp):.0f}%   p(median)={np.median(probs_i) if probs_i else -1:.3f}")
        print(f"  OVERALL ACCURACY: {100*(tp+tn)/n:.1f}%  (n={n})")
        print(f"  false 'finished' (would interrupt the speaker) FPR={100*fp/max(1,fp+tn):.1f}%")
        lat=np.array(lat); print(f"  inference: median={np.median(lat):.0f} ms min={lat.min():.0f} ms")
        res_all[os.path.basename(WAV)]=(tp+tn, n)
print("\n===== SUMMARY =====")
T=sum(v[0] for v in res_all.values()); N=sum(v[1] for v in res_all.values())
for k,v in res_all.items(): print(f"  {k}: {v[0]}/{v[1]} = {100*v[0]/v[1]:.1f}%")
print(f"  TOTAL: {T}/{N} = {100*T/max(1,N):.1f}%")
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
