"""Does T-one expose a usable per-phrase confidence signal (to route only doubtful phrases to a second recognizer)?

Requires `pip install -e T-one` (voicekit-team/T-one), onnxruntime, huggingface_hub.
Test clips (<key>-16k.wav / <key>-phone8k.wav) are read from $TESTAUDIO_DIR (default ./testaudio);
they are not shipped with this repo - supply your own recordings.
Writes results_confidence.json.
"""
import os, json, statistics as st
from pathlib import Path
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download
from tone.onnx_wrapper import StreamingCTCModel
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.decoder import GreedyCTCDecoder
from tone.pipeline import StreamingCTCPipeline
from tone import read_audio

AUDIO = Path(os.environ.get("TESTAUDIO_DIR", "testaudio"))

so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
model=StreamingCTCModel(ort.InferenceSession(hf_hub_download("t-tech/T-one","model.onnx"),sess_options=so))
dec=GreedyCTCDecoder(); CHUNK=StreamingCTCPipeline.CHUNK_SIZE; PAD=StreamingCTCPipeline.PADDING

def phrase_conf(audio):
    """Phrase confidence: mean probability of the chosen symbol over non-blank frames only."""
    a=np.pad(audio,(PAD,PAD)); ms=None; ls=None; sp=StreamingLogprobSplitter(); out=[]
    for i in range(0,len(a),CHUNK):
        c=a[i:i+CHUNK]; c=np.pad(c,(0,-len(c)%CHUNK))
        lp,ms=model.forward(c[None,:,None],ms)
        phs,ls=sp.forward(lp[0],ls)
        for ph in phs:
            L=ph.logprobs.astype(np.float32); best=L.argmax(-1)
            p=np.exp(L.max(-1))
            keep=best!=L.shape[-1]-1            # frames where the model emitted a symbol, not blank
            conf=float(p[keep].mean()) if keep.any() else float(p.mean())
            out.append((conf,dec.forward(L),len(L)))
    return out

res={}
for name in ["a53-phone8k","b79-phone8k","c193-phone8k","d285-phone8k"]:
    cs=phrase_conf(read_audio(AUDIO / f"{name}.wav"))
    v=sorted(c for c,_,_ in cs)
    res[name]={"phrases":len(v),"conf_min":round(v[0],3),"conf_p25":round(v[len(v)//4],3),
               "conf_median":round(v[len(v)//2],3),"conf_mean":round(st.mean(v),3),"conf_max":round(v[-1],3),
               "share_below_0.80":round(100*sum(1 for x in v if x<0.80)/len(v),1),
               "share_below_0.85":round(100*sum(1 for x in v if x<0.85)/len(v),1)}
    print(f"[{name:14}] phrases {len(v):3} | confidence: min {v[0]:.2f} p25 {v[len(v)//4]:.2f} median {v[len(v)//2]:.2f} max {v[-1]:.2f} "
          f"| below 0.80: {res[name]['share_below_0.80']:4.1f}% of phrases | below 0.85: {res[name]['share_below_0.85']:4.1f}%", flush=True)
    res[name]["_worst_lengths"]=[n for c,t,n in sorted(cs)[:3]]
Path("results_confidence.json").write_text(json.dumps(res,ensure_ascii=False,indent=2))
print("\nNote: a53 is a clean recording (22% disagreement vs reference), b79 is hard (44%).")
