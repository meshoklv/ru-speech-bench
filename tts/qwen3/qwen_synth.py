"""Synthesize the test phrases with Qwen3-TTS (voice cloning) and record RTF / peak VRAM.

Usage:
  python qwen_synth.py [--refs ./refs/refs.json] [--phrases phrases.json] [--out ./out/qwen3]

Needs: GPU, `pip install qwen-tts soundfile`, a refs.json (see common.load_refs).
The first reference voice is run on all phrases, any further voices on phrase "1" only.
RTF here is synth_time / audio_time (lower is faster; < 1 = faster than real time).
Writes <out>/metrics.json.
"""
import argparse
import json
import os
import time

import soundfile as sf
import torch

from common import GEN_KWARGS, HERE, MODEL_ID, ATTN, DEFAULT_REFS, load_model, load_refs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--refs", default=DEFAULT_REFS)
    ap.add_argument("--phrases", default=os.path.join(HERE, "phrases.json"))
    ap.add_argument("--out", default="./out/qwen3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.phrases, encoding="utf-8") as f:
        phrases = json.load(f)
    refs = load_refs(a.refs)

    print("loading", MODEL_ID, "attn=", ATTN, flush=True)
    t0 = time.time()
    model = load_model()
    print(f"LOAD_TIME {time.time()-t0:.2f}s", flush=True)
    print(f"VRAM_AFTER_LOAD_GB {torch.cuda.memory_allocated()/1e9:.2f}", flush=True)

    results = []
    for i, (refname, ref) in enumerate(refs.items()):
        tb = time.time()
        prompt = model.create_voice_clone_prompt(ref_audio=ref["file"], ref_text=ref["text"],
                                                 x_vector_only_mode=False)
        print(f"PROMPT_BUILD {refname} {time.time()-tb:.2f}s", flush=True)
        todo = list(phrases.items()) if i == 0 else [("1", phrases["1"])]
        for pid, text in todo:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            wavs, sr = model.generate_voice_clone(text=text, language="Russian",
                                                  voice_clone_prompt=prompt, **GEN_KWARGS)
            torch.cuda.synchronize()
            dt = time.time() - t1
            w = wavs[0]
            dur = len(w) / sr
            peak = torch.cuda.max_memory_allocated() / 1e9
            name = f"qwen3-{refname}-phrase{pid}.wav"
            sf.write(os.path.join(a.out, name), w, sr)
            print(f"RESULT {name} synth={dt:.2f}s audio={dur:.2f}s rtf={dt/dur:.3f} "
                  f"sr={sr} peak_alloc_gb={peak:.2f}", flush=True)
            results.append(dict(name=name, synth=dt, audio=dur, rtf=dt / dur, sr=sr, peak=peak))

    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("TOTAL_PEAK_ALLOC_GB %.2f" % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
