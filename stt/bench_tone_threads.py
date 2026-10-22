"""Effect of onnxruntime intra_op_num_threads on T-one speed (best of N repetitions).

Requires `pip install -e T-one` (voicekit-team/T-one), onnxruntime, huggingface_hub.
Test clips are read from $TESTAUDIO_DIR (default ./testaudio); they are not shipped - supply your own.
Writes results_threads.json.
"""
import os, time, sys, json
from pathlib import Path
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tone.onnx_wrapper import StreamingCTCModel
from tone.logprob_splitter import StreamingLogprobSplitter
from tone.decoder import GreedyCTCDecoder
from tone.pipeline import StreamingCTCPipeline
from tone import read_audio

AUDIO = Path(os.environ.get("TESTAUDIO_DIR", "testaudio"))

mp = hf_hub_download("t-tech/T-one", "model.onnx")
audio = read_audio(AUDIO / "a53-phone8k.wav"); dur = len(audio)/8000
REP = 4
res = {}
for n in [1, 2, 4, 8]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = n
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(mp, sess_options=so)
    pipe = StreamingCTCPipeline(StreamingCTCModel(sess), StreamingLogprobSplitter(), GreedyCTCDecoder())
    bw = bc = 1e9
    for _ in range(REP):
        tw, tc = time.perf_counter(), time.process_time()
        pipe.forward_offline(audio)
        bw = min(bw, time.perf_counter()-tw); bc = min(bc, time.process_time()-tc)
    res[n] = {"wall": round(bw,2), "cpu": round(bc,2), "rtf": round(bw/dur,3), "speedup": round(dur/bw,2), "cpu_rtf": round(bc/dur,3)}
    print(f"intra_op_num_threads={n}: best wall {bw:.2f}s (RTF {bw/dur:.3f}, x{dur/bw:.2f}) | CPU time {bc:.2f}s (CPU-RTF {bc/dur:.3f})", flush=True)
Path("results_threads.json").write_text(json.dumps({"audio_s": round(dur,2), "reps": REP, "res": res}, indent=2))
print("audio:", round(dur,2), "s; loadavg:", open("/proc/loadavg").read().split()[:3])
