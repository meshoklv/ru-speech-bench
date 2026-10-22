"""How many concurrent streaming lines can ONE process (one shared LM + one ORT session) sustain?

Requires `pip install -e T-one` (voicekit-team/T-one), onnxruntime, huggingface_hub.
Test clips are read from $TESTAUDIO_DIR (default ./testaudio); they are not shipped - supply your own.
Uses the beam-search decoder (kenlm). Writes results_parallel.json.
"""
import os, time, json, threading, resource, statistics as st
from pathlib import Path
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download
from tone.onnx_wrapper import StreamingCTCModel
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.decoder import BeamSearchCTCDecoder
from tone.pipeline import StreamingCTCPipeline
from tone import read_audio

AUDIO = Path(os.environ.get("TESTAUDIO_DIR", "testaudio"))

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
sess=ort.InferenceSession(hf_hub_download("t-tech/T-one","model.onnx"),sess_options=so)
dec=BeamSearchCTCDecoder.from_hugging_face()          # one language model shared by all lines
model=StreamingCTCModel(sess)                          # one ORT session shared by all lines
print(f"loaded, RSS {rss():.0f} MB (one LM shared by all lines)", flush=True)

CHUNK,PAD=StreamingCTCPipeline.CHUNK_SIZE,StreamingCTCPipeline.PADDING
audio=read_audio(AUDIO / "a53-phone8k.wav"); DUR=len(audio)/8000
audio=np.pad(audio,(PAD,PAD))

def line(res,idx):
    pipe=StreamingCTCPipeline(model,StreamingLogprobSplitter(),dec)  # own splitter = own state
    state=None; cts=[]
    t0=time.perf_counter()
    for i in range(0,len(audio),CHUNK):
        c=audio[i:i+CHUNK]; c=np.pad(c,(0,-len(c)%CHUNK))
        t=time.perf_counter(); _,state=pipe.forward(c,state); cts.append((time.perf_counter()-t)*1000)
    cts.sort()
    res[idx]={"total_s":round(time.perf_counter()-t0,2),"p50":round(cts[len(cts)//2],1),
              "p95":round(cts[int(len(cts)*.95)],1),"max":round(cts[-1],1),
              "over300":round(100*sum(1 for x in cts if x>300)/len(cts),1)}

out={}
for n in [1,2,4]:
    res={}; ths=[threading.Thread(target=line,args=(res,i)) for i in range(n)]
    t0=time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    wall=time.perf_counter()-t0
    worst=max(r["p95"] for r in res.values()); over=max(r["over300"] for r in res.values())
    slow=max(r["total_s"] for r in res.values())
    out[n]={"wall_s":round(wall,2),"audio_s":round(DUR,1),"slowest_line_s":round(slow,2),
            "worst_p95_ms":worst,"worst_over300_pct":over,"rss_mb":round(rss()),
            "keeps_realtime": slow < DUR}
    print(f"[{n} concurrent lines] slowest line {slow:6.2f}s for {DUR:.0f}s of audio "
          f"| worst p95 per chunk {worst:6.1f} ms | chunks over 300 ms {over:4.1f}% "
          f"| {'KEEPS' if slow<DUR else 'MISSES'} real time | RSS {rss():.0f} MB", flush=True)
Path("results_parallel.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
print("loadavg:",open("/proc/loadavg").read().split()[:3])
