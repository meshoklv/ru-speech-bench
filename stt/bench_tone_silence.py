"""Effect of the splitter silence threshold (MIN_SILENCE_DURATION) on latency and on phrase splitting.

Requires `pip install -e T-one` (voicekit-team/T-one), onnxruntime, huggingface_hub.
Test clips are read from $TESTAUDIO_DIR (default ./testaudio); they are not shipped - supply your own.
Needs two reference transcripts in ./ref/ (FILES below; produced by any other recognizer).
The "WER" printed is disagreement with that reference, not true WER. Writes results_silence.json.
"""
import os, time, json, re, statistics as st
from pathlib import Path
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download
from tone.onnx_wrapper import StreamingCTCModel
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.decoder import BeamSearchCTCDecoder
from tone.pipeline import StreamingCTCPipeline
from tone import read_audio

AUDIO = Path(os.environ.get("TESTAUDIO_DIR", "testaudio"))

def norm(s): return re.sub(r'[^а-я\s]',' ',s.lower().replace('ё','е')).split()
def wer(ref,hyp):
    r,h=norm(ref),norm(hyp)
    d=[[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0]=i
    for j in range(len(h)+1): d[0][j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(r[i-1]!=h[j-1]))
    return d[len(r)][len(h)]/max(1,len(r))

so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
sess=ort.InferenceSession(hf_hub_download("t-tech/T-one","model.onnx"),sess_options=so)
dec=BeamSearchCTCDecoder.from_hugging_face()
CHUNK,PAD=StreamingCTCPipeline.CHUNK_SIZE,StreamingCTCPipeline.PADDING
FR=StreamingCTCModel.FRAME_SIZE
FILES={"a53-phone8k":"ref/a53-phone8k.ref.txt","b79-phone8k":"ref/b79-phone8k.ref.txt"}
DEFAULT=StreamingLogprobSplitter.MIN_SILENCE_DURATION
print(f"default MIN_SILENCE_DURATION={DEFAULT} frames = {DEFAULT*FR:.2f} s of silence\n")
out={}
for msd in [20,16,14,12,10,8]:
    row={}
    for name,refp in FILES.items():
        StreamingLogprobSplitter.MIN_SILENCE_DURATION=msd
        pipe=StreamingCTCPipeline(StreamingCTCModel(sess),StreamingLogprobSplitter(),dec)
        a=read_audio(AUDIO / f"{name}.wav"); dur=len(a)/8000
        a=np.pad(a,(PAD,PAD)); state=None; lags=[]; texts=[]
        for i in range(0,len(a),CHUNK):
            c=a[i:i+CHUNK]; c=np.pad(c,(0,-len(c)%CHUNK))
            ph,state=pipe.forward(c,state)
            if ph:
                pos=(i+CHUNK)/8000-PAD/8000
                for p in ph: lags.append(pos-p.end_time); texts.append(p.text)
        tail,_=pipe.finalize(state); texts+= [p.text for p in tail]
        txt=" ".join(texts); e=wer(open(refp).read(),txt)
        row[name]={"lag_mean":round(st.mean(lags),2),"lag_max":round(max(lags),2),
                   "phrases":len(texts),"wer":round(e,3),"chars":len(txt)}
        print(f"  MIN_SILENCE={msd:2} ({msd*FR:.2f}c) {name:14} mean lag {st.mean(lags):.2f}s max {max(lags):.2f}s | phrases {len(texts):3} | disagreement {e*100:.1f}%", flush=True)
    out[msd]=row
StreamingLogprobSplitter.MIN_SILENCE_DURATION=DEFAULT
Path("results_silence.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
