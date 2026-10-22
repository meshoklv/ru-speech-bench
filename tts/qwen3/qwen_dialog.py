"""Synthesize a two-speaker dialogue (jobs.json) with Qwen3-TTS voice cloning.

Usage:
  python qwen_dialog.py [--jobs jobs.json] [--refs ./refs/refs.json] [--out ./out/dialog] [job names...]

jobs.json: [{"name","ref","seed","text"}, ...]; "ref" is a key in refs.json
(the bundled jobs.json expects two voices named voiceA and voiceB).
All lines of one speaker share ONE clone prompt and identical sampling
parameters, otherwise the voice drifts between lines; only the seed differs.
Needs: GPU, `pip install qwen-tts soundfile`.
"""
import argparse
import json
import os
import time

import soundfile as sf

from common import GEN_KWARGS, HERE, DEFAULT_REFS, load_model, load_refs, make_prompts, seed_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--jobs", default=os.path.join(HERE, "jobs.json"))
    ap.add_argument("--refs", default=DEFAULT_REFS)
    ap.add_argument("--out", default="./out/dialog")
    ap.add_argument("only", nargs="*", help="regenerate only these job names")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.jobs, encoding="utf-8") as f:
        jobs = json.load(f)
    if a.only:
        jobs = [j for j in jobs if j["name"] in set(a.only)]
    refs = load_refs(a.refs)

    model = load_model()
    prompts = make_prompts(model, refs)
    for j in jobs:
        seed_all(j["seed"])
        t = time.time()
        wavs, sr = model.generate_voice_clone(text=j["text"], language="Russian",
                                              voice_clone_prompt=prompts[j["ref"]], **GEN_KWARGS)
        dt = time.time() - t
        w = wavs[0]
        dur = len(w) / sr
        sf.write(os.path.join(a.out, j["name"] + ".wav"), w, sr)
        print(f"RESULT {j['name']}.wav ref={j['ref']} seed={j['seed']} "
              f"synth={dt:.2f}s audio={dur:.2f}s rtf={dt/dur:.3f}", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
