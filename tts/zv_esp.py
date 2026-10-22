"""Synthesize the dialogue (jobs JSON) with ESpeech-TTS (F5-TTS architecture + RUAccent).

Usage:
  python zv_esp.py [names_comma_separated] [seed_override] \
        [--jobs qwen3/jobs.json] [--refs ./refs/refs.json] [--out ./out/zv_esp]

Reference clips must be <= 12 s (model card). jobs JSON: [{"name","ref","seed","text"}].
Optional seed override appends _s<seed> to the file name. Needs: GPU,
`pip install f5-tts ruaccent soundfile`, ESpeech weights (see espeech/espeech_synth.py).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "espeech"))

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import common  # noqa: E402

common.patch_audio_io()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("names", nargs="?", default="", help="comma-separated job names (default: all)")
    ap.add_argument("seed", nargs="?", type=int, default=None, help="override the per-job seed")
    ap.add_argument("--jobs", default=os.path.join(HERE, "qwen3", "jobs.json"))
    ap.add_argument("--refs", default=common.DEFAULT_REFS)
    ap.add_argument("--out", default="./out/zv_esp")
    ap.add_argument("--ckpt", default=common.DEFAULT_CKPT)
    ap.add_argument("--vocab", default=common.DEFAULT_VOCAB)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.jobs, encoding="utf-8") as f:
        jobs = json.load(f)
    if a.names:
        jobs = [j for j in jobs if j["name"] in a.names.split(",")]
    refs = common.load_refs(a.refs)

    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    model, vocoder, acc = common.load_espeech(a.ckpt, a.vocab)
    accent = common.make_accent(acc)
    pre = {rn: preprocess_ref_audio_text(r["file"], accent(r["text"])) for rn, r in refs.items()}

    for j in jobs:
        s = a.seed if a.seed is not None else j["seed"]
        common.seed_all(s)
        ra, rt = pre[j["ref"]]
        g = accent(j["text"])
        t = time.time()
        wave, sr, _ = infer_process(ra, rt, g, model, vocoder, cross_fade_duration=0.15,
                                    nfe_step=32, speed=1.0)
        dt = time.time() - t
        suff = "" if a.seed is None else f"_s{s}"
        sf.write(os.path.join(a.out, j["name"] + suff + ".wav"), wave, sr)
        print(f"RESULT {j['name']}{suff}.wav ref={j['ref']} seed={s} synth={dt:.2f}s "
              f"audio={len(wave)/sr:.2f}s rtf={dt/(len(wave)/sr):.3f}", flush=True)
        print(f"  ACCENTED: {g}", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
