"""Generate several alternative takes of ONE dialogue line with Qwen3-TTS (different seeds).

Usage:
  python k3fix.py --job k3 [--seeds 3001,3002,3003,3004] \
        [--jobs qwen3/jobs.json] [--refs ./refs/refs.json] [--out ./out/zv_qwen]

Pick the take without reference leakage (see check_zv.py). Needs: GPU,
`pip install qwen-tts soundfile`.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "qwen3"))

import soundfile as sf  # noqa: E402
from common import GEN_KWARGS, DEFAULT_REFS, load_model, load_refs, make_prompts, seed_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--seeds", default="3001,3002,3003,3004")
    ap.add_argument("--jobs", default=os.path.join(HERE, "qwen3", "jobs.json"))
    ap.add_argument("--refs", default=DEFAULT_REFS)
    ap.add_argument("--out", default="./out/zv_qwen")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.jobs, encoding="utf-8") as f:
        j = {x["name"]: x for x in json.load(f)}[a.job]
    refs = {j["ref"]: load_refs(a.refs)[j["ref"]]}
    model = load_model()
    prompt = make_prompts(model, refs)[j["ref"]]
    for s in [int(x) for x in a.seeds.split(",")]:
        seed_all(s)
        wavs, sr = model.generate_voice_clone(text=j["text"], language="Russian",
                                              voice_clone_prompt=prompt, **GEN_KWARGS)
        sf.write(os.path.join(a.out, f"{a.job}_s{s}.wav"), wavs[0], sr)
        print(f"CAND {a.job}_s{s}.wav audio={len(wavs[0])/sr:.2f}s", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
