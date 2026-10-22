"""Offline T-one run with beam search + kenlm language model, 1 onnxruntime thread.

Requires `pip install -e T-one` (voicekit-team/T-one), onnxruntime, huggingface_hub.
Test clips (<key>-16k.wav / <key>-phone8k.wav) are read from $TESTAUDIO_DIR (default ./testaudio);
they are not shipped with this repo - supply your own recordings.
Needs kenlm bindings and the 5.4 GB kenlm.bin from the HF repo. Writes results_beam.json.
"""
import os, time, json, resource
from pathlib import Path
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tone.onnx_wrapper import StreamingCTCModel
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.decoder import BeamSearchCTCDecoder
from tone.pipeline import StreamingCTCPipeline
from tone import read_audio

AUDIO = Path(os.environ.get("TESTAUDIO_DIR", "testaudio"))

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
t0=time.perf_counter()
sess = ort.InferenceSession(hf_hub_download("t-tech/T-one","model.onnx"), sess_options=so)
pipe = StreamingCTCPipeline(StreamingCTCModel(sess), StreamingLogprobSplitter(), BeamSearchCTCDecoder.from_hugging_face())
print(f"### load (model + LM): {time.perf_counter()-t0:.1f} s, RSS {rss():.0f} MB", flush=True)
runs={}
for key in ["a53","b79","c193","d285"]:
    for v in ["16k","phone8k"]:
        a = read_audio(AUDIO / f"{key}-{v}.wav"); d=len(a)/8000
        tw, tc = time.perf_counter(), time.process_time()
        ph = pipe.forward_offline(a)
        w, c = time.perf_counter()-tw, time.process_time()-tc
        txt = " ".join(p.text for p in ph)
        runs[f"{key}-{v}"]={"audio_s":round(d,2),"proc_s":round(w,2),"cpu_s":round(c,2),
            "rtf":round(w/d,4),"speedup":round(d/w,2),"cpu_rtf":round(c/d,4),
            "n_phrases":len(ph),"chars":len(txt),"text_chars":len(txt)}
        print(f"[{key}-{v}] audio {d:6.1f}s | wall {w:7.2f}s | CPU {c:7.2f}s | RTF {w/d:.3f} | x{d/w:.2f} | phrases {len(ph):3} | chars {len(txt)} | RSS {rss():.0f} MB", flush=True)
Path("results_beam.json").write_text(json.dumps({"peak_rss_mb":round(rss()),"runs":runs},ensure_ascii=False,indent=2))
print("loadavg:", open("/proc/loadavg").read().split()[:3])
