"""Re-generate selected dialogue lines with several seeds (voice and params unchanged).

Used to work around "reference leakage" (the model prepends/continues words of the
reference clip): generate candidates, check them with check_dialog.py, keep the clean one.

Usage:
  python qwen_fix.py k1,k6 3001,3002,3003 [--jobs jobs.json] [--refs ./refs/refs.json] [--out ./out/dialog]
Needs: GPU, `pip install qwen-tts soundfile`.
"""
import argparse
import json
import os

import soundfile as sf

from common import GEN_KWARGS, HERE, DEFAULT_REFS, load_model, load_refs, make_prompts, seed_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", help="comma-separated job names")
    ap.add_argument("seeds", help="comma-separated integer seeds")
    ap.add_argument("--jobs", default=os.path.join(HERE, "jobs.json"))
    ap.add_argument("--refs", default=DEFAULT_REFS)
    ap.add_argument("--out", default="./out/dialog")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.jobs, encoding="utf-8") as f:
        jobs = {j["name"]: j for j in json.load(f)}
    seeds = [int(x) for x in a.seeds.split(",")]
    model = load_model()
    prompts = make_prompts(model, load_refs(a.refs))
    for name in a.names.split(","):
        j = jobs[name]
        for s in seeds:
            seed_all(s)
            wavs, sr = model.generate_voice_clone(text=j["text"], language="Russian",
                                                  voice_clone_prompt=prompts[j["ref"]], **GEN_KWARGS)
            sf.write(os.path.join(a.out, f"{name}_s{s}.wav"), wavs[0], sr)
            print(f"CAND {name}_s{s}.wav audio={len(wavs[0])/sr:.2f}s", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
