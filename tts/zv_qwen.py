"""Synthesize the dialogue (jobs JSON) with Qwen3-TTS, one clone prompt per speaker.

Usage:
  python zv_qwen.py [names_comma_separated] [seed_override] \
        [--jobs qwen3/jobs.json] [--refs ./refs/refs.json] [--out ./out/zv_qwen]

Same as qwen3/qwen_dialog.py but with an optional seed override that appends _s<seed>
to the file name (to produce alternative takes). Needs: GPU, `pip install qwen-tts soundfile`.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "qwen3"))

import soundfile as sf  # noqa: E402
from common import GEN_KWARGS, DEFAULT_REFS, load_model, load_refs, make_prompts, seed_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="?", default="", help="comma-separated job names (default: all)")
    ap.add_argument("seed", nargs="?", type=int, default=None, help="override the per-job seed")
    ap.add_argument("--jobs", default=os.path.join(HERE, "qwen3", "jobs.json"))
    ap.add_argument("--refs", default=DEFAULT_REFS)
    ap.add_argument("--out", default="./out/zv_qwen")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.jobs, encoding="utf-8") as f:
        jobs = json.load(f)
    if a.names:
        jobs = [j for j in jobs if j["name"] in a.names.split(",")]
    model = load_model()
    prompts = make_prompts(model, load_refs(a.refs))
    for j in jobs:
        s = a.seed if a.seed is not None else j["seed"]
        seed_all(s)
        t = time.time()
        wavs, sr = model.generate_voice_clone(text=j["text"], language="Russian",
                                              voice_clone_prompt=prompts[j["ref"]], **GEN_KWARGS)
        dt = time.time() - t
        w = wavs[0]
        suff = "" if a.seed is None else f"_s{s}"
        sf.write(os.path.join(a.out, j["name"] + suff + ".wav"), w, sr)
        print(f"RESULT {j['name']}{suff}.wav ref={j['ref']} seed={s} synth={dt:.2f}s "
              f"audio={len(w)/sr:.2f}s rtf={dt/(len(w)/sr):.3f}", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
