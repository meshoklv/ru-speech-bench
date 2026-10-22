"""Smart Turn v3: controlled A/B on the SAME context.

  A) audio ends at a natural phrase boundary (followed by a long pause);
  B) the same audio cut 0.7 s EARLIER, i.e. in the middle of a word.
A useful end-of-turn model should give clearly higher p(finished) for A than B.
Reports median p, share of p>0.5, the share of pairs with A>B and the ROC-AUC.

Usage:
    pip install onnxruntime transformers silero-vad torch numpy
    SMART_TURN_MODEL=./smart-turn-v3.2-cpu.onnx python3 st_ab.py rec1.wav rec2.wav ...
Input: your own 16 kHz-resampled recordings of continuous speech with pauses.
"""
import sys, os, time, numpy as np, onnxruntime as ort, torch
MODEL = os.environ.get("SMART_TURN_MODEL", "./smart-turn-v3.2-cpu.onnx")
torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import WhisperFeatureExtractor
from audio_utils import truncate_audio_to_last_n_seconds
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
so = ort.SessionOptions(); so.intra_op_num_threads=4; so.inter_op_num_threads=1
sess = ort.InferenceSession(MODEL, sess_options=so); IN=sess.get_inputs()[0].name
fe = WhisperFeatureExtractor(chunk_length=8)
def p_of(a):
    a = truncate_audio_to_last_n_seconds(a, 8)
    f = fe(a, sampling_rate=16000, return_tensors="np", padding="max_length",
           max_length=8*16000, truncation=True, do_normalize=True).input_features.astype(np.float32)
    x = float(np.ravel(sess.run(None, {IN: f})[0])[0]); return x if 0<=x<=1 else 1/(1+np.exp(-x))
SR=16000; vad = load_silero_vad()
PA, PB = [], []
for WAV in sys.argv[1:]:
    wav = read_audio(WAV, sampling_rate=16000); a = wav.numpy()
    ts = get_speech_timestamps(wav, vad, sampling_rate=16000, return_seconds=True)
    for i, s in enumerate(ts[:-1]):
        gap = ts[i+1]['start'] - s['end']
        if gap < 1.0 or s["end"]-s["start"] < 1.5: continue
        eA = s['end'] + 0.2                      # natural end + 200 ms of silence
        eB = s['end'] - 0.7                      # cut in the middle of a word
        segA = a[max(0,int((eA-8)*SR)):int(eA*SR)]
        segB = a[max(0,int((eB-8)*SR)):int(eB*SR)]
        pa, pb = p_of(segA), p_of(segB)
        PA.append(pa); PB.append(pb)
        print(f"  {os.path.basename(WAV)[:12]} @{s['end']:6.1f}s  A(phrase end)={pa:.3f}   B(mid-word cut)={pb:.3f}   {'✓ A>B' if pa>pb else '✗'}")
PA, PB = np.array(PA), np.array(PB)
print(f"\npairs: {len(PA)}")
print(f"A (speaker finished)   : median p={np.median(PA):.3f}  share p>0.5 = {100*(PA>0.5).mean():.0f}%")
print(f"B (cut mid-word)       : median p={np.median(PB):.3f}  share p>0.5 = {100*(PB>0.5).mean():.0f}%")
print(f"A>B in {100*(PA>PB).mean():.0f}% of pairs")
# AUC over these pairs
lab = np.r_[np.ones(len(PA)), np.zeros(len(PB))]; sc = np.r_[PA, PB]
o = np.argsort(sc); r = np.empty(len(sc)); r[o] = np.arange(1, len(sc)+1)
n1, n0 = lab.sum(), (1-lab).sum()
print(f"ROC-AUC (A vs B) = {(r[lab==1].sum() - n1*(n1+1)/2)/(n1*n0):.3f}   (0.5 = coin flip, 1.0 = perfect)")
