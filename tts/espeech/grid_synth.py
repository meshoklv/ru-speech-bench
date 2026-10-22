"""Grid over ESpeech expressiveness settings (cfg_strength, nfe_step, sway sampling, speed)
on conversational reference clips: can the knobs close the gap to the reference's liveliness?

Usage:
  python grid_synth.py --refs ./refs/live_refs.json --voices f1,m1 [--out ./out/grid]

refs JSON format: see common.load_refs; --voices selects keys from it.
Needs: GPU, `pip install f5-tts ruaccent soundfile`, ESpeech weights (see espeech_synth.py).
Fixed seed, so only the settings differ between files. Writes grid.json.
"""
import argparse
import json
import os
import time

import soundfile as sf
import torch

import common
common.patch_audio_io()

TEXTS = {"p2": "Ваша запись уже готова, я звоню уточнить: приходите сами или перенести на другой день?",
         "p2r": "Ваша запись, ну, уже готова. Я звоню уточнить - приходите сами, или, может, переносим на другой день?"}
GRID = [("base",      dict(nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, speed=1.0)),
        ("cfg3",      dict(nfe_step=32, cfg_strength=3.0, sway_sampling_coef=-1.0, speed=1.0)),
        ("cfg4",      dict(nfe_step=32, cfg_strength=4.0, sway_sampling_coef=-1.0, speed=1.0)),
        ("nfe64",     dict(nfe_step=64, cfg_strength=2.0, sway_sampling_coef=-1.0, speed=1.0)),
        ("sway-05",   dict(nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-0.5, speed=1.0)),
        ("sway0",     dict(nfe_step=32, cfg_strength=2.0, sway_sampling_coef=0.0, speed=1.0)),
        ("slow92",    dict(nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, speed=0.92)),
        ("cfg3nfe64", dict(nfe_step=64, cfg_strength=3.0, sway_sampling_coef=-1.0, speed=0.95))]
SEED = 2002


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--refs", default="./refs/live_refs.json")
    ap.add_argument("--voices", required=True, help="comma-separated keys of the refs JSON")
    ap.add_argument("--out", default="./out/grid")
    ap.add_argument("--ckpt", default=common.DEFAULT_CKPT)
    ap.add_argument("--vocab", default=common.DEFAULT_VOCAB)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    refs = common.load_refs(a.refs)
    use = a.voices.split(",")

    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    model, vocoder, acc = common.load_espeech(a.ckpt, a.vocab)
    accent = common.make_accent(acc)
    prep = {k: preprocess_ref_audio_text(refs[k]["file"], accent(refs[k]["text"])) for k in use}

    res = []
    for gname, kw in GRID:
        for tname, text in TEXTS.items():
            g = accent(text)
            for k in use:
                ra, rt = prep[k]
                common.seed_all(SEED)
                torch.cuda.synchronize()
                t1 = time.time()
                w, sr, _ = infer_process(ra, rt, g, model, vocoder, cross_fade_duration=0.15, **kw)
                torch.cuda.synchronize()
                dt = time.time() - t1
                dur = len(w) / sr
                nm = f"grid-{k}-{tname}-{gname}.wav"
                sf.write(os.path.join(a.out, nm), w, sr)
                print(f"RESULT {nm} synth={dt:.2f}s audio={dur:.2f}s rtf={dt/dur:.3f}", flush=True)
                res.append(dict(name=nm, ref=k, text=tname, grid=gname, rtf=dt / dur, audio=dur, **kw))
        with open(os.path.join(a.out, "grid.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
