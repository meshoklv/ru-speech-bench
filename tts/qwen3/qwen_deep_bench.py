"""Deep speed analysis of Qwen3-TTS: cold vs warm run, GPU vs CPU placement,
number of talker forward passes, time per pass, GPU utilisation, time to first pass.

Usage:
  python qwen_deep_bench.py --voice voiceA [--refs ./refs/refs.json] [--out ./out/deep_bench]

Needs: NVIDIA GPU + nvidia-smi on PATH, `pip install qwen-tts soundfile`.
Runs 1 cold + 5 warm generations of the same phrase and writes deep_bench.json.
RTF = synth_time / audio_time (lower is faster).
"""
import argparse
import json
import os
import statistics
import subprocess
import threading
import time

import soundfile as sf
import torch

from common import GEN_KWARGS, DEFAULT_REFS, load_model, load_refs

TEXT = "Здравствуйте! Меня зовут Кирилл, барбершоп «Ромашка». Вам удобно сейчас говорить?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--refs", default=DEFAULT_REFS)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", default="./out/deep_bench")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ref = load_refs(a.refs)[a.voice]

    print("== ENVIRONMENT ==", flush=True)
    print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(), flush=True)
    print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0), flush=True)

    t0 = time.time()
    m = load_model()
    print(f"LOAD {time.time()-t0:.2f}s", flush=True)

    # where do the weights actually live
    inner = getattr(m, "model", m)
    devs, dts = {}, {}
    for _, p in inner.named_parameters():
        devs[str(p.device)] = devs.get(str(p.device), 0) + p.numel()
        dts[str(p.dtype)] = dts.get(str(p.dtype), 0) + p.numel()
    print("== WEIGHT PLACEMENT ==", flush=True)
    for d, n in sorted(devs.items(), key=lambda x: -x[1]):
        print(f"  {d}: {n/1e9:.3f} B parameters", flush=True)
    print("  dtypes:", {k: f"{v/1e9:.3f}B" for k, v in dts.items()}, flush=True)

    # Count forward passes of the talker via a standard PyTorch hook.
    # (Do NOT monkey-patch talker.forward: generate() inspects its signature and fails.)
    talker = None
    for path in ["talker", "model.talker", "thinker"]:
        o = inner
        try:
            for part in path.split("."):
                o = getattr(o, part)
            talker = o
            print("TALKER:", path, type(o).__name__, flush=True)
            break
        except AttributeError:
            pass
    if talker is None:
        print("TALKER not found; children:", [n for n, _ in inner.named_children()], flush=True)

    state = {"n": 0, "t_first": None, "t_last": None, "t0": None}
    if talker is not None:
        def _pre_hook(mod, args, kwargs=None):
            if state["t0"] is not None:
                state["n"] += 1
                now = time.time()
                if state["t_first"] is None:
                    state["t_first"] = now - state["t0"]
                state["t_last"] = now - state["t0"]
            return None
        talker.register_forward_pre_hook(_pre_hook, with_kwargs=True)

    samp = {"run": False, "util": [], "mem": []}

    def sampler():
        while samp["run"]:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
                u, mm = o.split(",")[:2]
                samp["util"].append(int(u))
                samp["mem"].append(int(mm))
            except Exception:
                pass
            time.sleep(0.25)

    prompt = m.create_voice_clone_prompt(ref_audio=ref["file"], ref_text=ref["text"],
                                         x_vector_only_mode=False)
    rows = []
    print("== RUNS (1 cold + 5) ==", flush=True)
    for i in range(6):
        state.update(n=0, t_first=None, t_last=None)
        samp["util"] = []
        samp["mem"] = []
        samp["run"] = True
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        torch.cuda.synchronize()
        state["t0"] = time.time()
        wavs, sr = m.generate_voice_clone(text=TEXT, language="Russian",
                                          voice_clone_prompt=prompt, **GEN_KWARGS)
        torch.cuda.synchronize()
        dt = time.time() - state["t0"]
        samp["run"] = False
        th.join(timeout=2)
        dur = len(wavs[0]) / sr
        if i == 0:
            sf.write(os.path.join(a.out, "bench-cold.wav"), wavs[0], sr)
        um = max(samp["util"]) if samp["util"] else None
        ua = round(sum(samp["util"]) / len(samp["util"]), 1) if samp["util"] else None
        mm = max(samp["mem"]) if samp["mem"] else None
        tag = "COLD" if i == 0 else f"run{i}"
        print(f"RUN {tag} synth={dt:.3f}s audio={dur:.2f}s rtf={dt/dur:.3f} passes={state['n']} "
              f"first_pass={state['t_first']} last_pass={state['t_last']} "
              f"gpu_util_max={um} gpu_util_avg={ua} gpu_mem_max={mm}MiB", flush=True)
        rows.append(dict(i=i, dt=dt, dur=dur, rtf=dt / dur, n=state["n"], t_first=state["t_first"],
                         t_last=state["t_last"], util_max=um, mem_max=mm))

    warm = rows[1:]
    print("== SUMMARY (cold run excluded) ==", flush=True)
    print("MEDIAN synth  = %.3f s" % statistics.median(r["dt"] for r in warm), flush=True)
    print("MEDIAN rtf    = %.3f" % statistics.median(r["rtf"] for r in warm), flush=True)
    print("MEDIAN audio  = %.2f s" % statistics.median(r["dur"] for r in warm), flush=True)
    if warm[0]["t_first"] is not None:
        print("MEDIAN time to first model pass = %.3f s" % statistics.median(r["t_first"] for r in warm), flush=True)
        print("MEDIAN number of passes = %d" % statistics.median(r["n"] for r in warm), flush=True)
        md = statistics.median((r["t_last"] - r["t_first"]) / max(r["n"] - 1, 1) for r in warm)
        print("MEDIAN time per pass = %.1f ms" % (md * 1000), flush=True)
        print("cold run slower than warm by %.1f%%" %
              ((rows[0]["rtf"] / statistics.median(r["rtf"] for r in warm) - 1) * 100), flush=True)
    with open(os.path.join(a.out, "deep_bench.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
