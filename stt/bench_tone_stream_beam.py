"""Streaming T-one run, intra_op_num_threads=1: per-chunk latency for 300 ms chunks (beam search + kenlm decoder).

Requires `pip install -e T-one` (voicekit-team/T-one), onnxruntime, huggingface_hub.
Test clips are read from $TESTAUDIO_DIR (default ./testaudio); they are not shipped - supply your own.
Writes results_stream_beam.json.
"""
import os, time, json, resource, statistics as st
from pathlib import Path
import numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download
from tone.onnx_wrapper import StreamingCTCModel
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.decoder import BeamSearchCTCDecoder
from tone.pipeline import StreamingCTCPipeline
from tone import read_audio

AUDIO = Path(os.environ.get("TESTAUDIO_DIR", "testaudio"))

so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
sess = ort.InferenceSession(hf_hub_download("t-tech/T-one","model.onnx"), sess_options=so)
pipe = StreamingCTCPipeline(StreamingCTCModel(sess), StreamingLogprobSplitter(), BeamSearchCTCDecoder.from_hugging_face())
CHUNK, PAD = StreamingCTCPipeline.CHUNK_SIZE, StreamingCTCPipeline.PADDING
out={}
for name in ["a53-phone8k","b79-phone8k","c193-phone8k"]:
    audio = read_audio(AUDIO / f"{name}.wav"); dur=len(audio)/8000
    audio = np.pad(audio,(PAD,PAD)); state=None; cts=[]; lags=[]; first=None
    t0=time.perf_counter()
    for i in range(0,len(audio),CHUNK):
        c=audio[i:i+CHUNK]; c=np.pad(c,(0,-len(c)%CHUNK))
        t=time.perf_counter(); ph,state=pipe.forward(c,state); cts.append((time.perf_counter()-t)*1000)
        if ph:
            pos=(i+CHUNK)/8000-PAD/8000
            for p in ph: lags.append(pos-p.end_time)
            if first is None: first={"wall_s":round(time.perf_counter()-t0,2),"lag_s":round(pos-ph[0].end_time,2)}
    cts.sort(); tot=time.perf_counter()-t0
    r={"audio_s":round(dur,1),"proc_s":round(tot,2),"rtf":round(tot/dur,3),"speedup":round(dur/tot,2),
       "chunks":len(cts),"mean_ms":round(st.mean(cts),1),"p50_ms":round(cts[len(cts)//2],1),
       "p95_ms":round(cts[int(len(cts)*0.95)],1),"p99_ms":round(cts[int(len(cts)*0.99)],1),"max_ms":round(cts[-1],1),
       "over300ms_pct":round(100*sum(1 for x in cts if x>300)/len(cts),1),
       "first_phrase":{k:v for k,v in first.items() if k!="text"},"lag_mean_s":round(st.mean(lags),2),"lag_max_s":round(max(lags),2),
       "peak_rss_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)}
    out[name]=r
    print(f"[{name}] audio {r['audio_s']}s, chunks {r['chunks']}, RTF {r['rtf']} (x{r['speedup']})")
    print(f"  per 300 ms chunk: mean {r['mean_ms']} | p50 {r['p50_ms']} | p95 {r['p95_ms']} | p99 {r['p99_ms']} | max {r['max_ms']} ms; chunks over 300 ms: {r['over300ms_pct']}%")
    print(f"  first phrase after {first['wall_s']}s, lag after phrase end {first['lag_s']}s | mean lag {r['lag_mean_s']}s, max {r['lag_max_s']}s | peakRSS {r['peak_rss_mb']} MB", flush=True)
Path("results_stream_beam.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
print("loadavg:", open("/proc/loadavg").read().split()[:3])
