"""Qwen3-TTS speed benchmark: 3 phrase lengths x 3 repeats, RTF and peak VRAM.

Usage:
  python qwen_bench.py --voice voiceA [--refs ./refs/refs.json] [--out ./out/qwen3_bench]

Needs: GPU, `pip install qwen-tts soundfile`, a refs.json with the chosen voice.
RTF = synth_time / audio_time (lower is faster; < 1 = faster than real time).
For a clean measurement run it on an otherwise idle GPU.
"""
import argparse
import os
import time

import soundfile as sf
import torch

from common import GEN_KWARGS, DEFAULT_REFS, load_model, load_refs

CASES = [("short", "Здравствуйте!"),
         ("greet", "Здравствуйте! Меня зовут Анна, салон «Ромашка»."),
         ("full", "Здравствуйте! Меня зовут Анна, салон «Ромашка». Звоню по поводу вашей записи.")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--refs", default=DEFAULT_REFS)
    ap.add_argument("--voice", required=True, help="key in refs.json")
    ap.add_argument("--out", default="./out/qwen3_bench")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ref = load_refs(a.refs)[a.voice]
    t0 = time.time()
    model = load_model()
    print(f"LOAD_TIME {time.time()-t0:.2f}s (weights cached)", flush=True)
    print(f"VRAM_WEIGHTS_GB {torch.cuda.memory_allocated()/1e9:.2f}", flush=True)
    prompt = model.create_voice_clone_prompt(ref_audio=ref["file"], ref_text=ref["text"],
                                             x_vector_only_mode=False)
    print("GPU_FREE_MIB_BEFORE", torch.cuda.mem_get_info()[0] // 2**20, flush=True)
    for name, text in CASES:
        for i in range(a.repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            wavs, sr = model.generate_voice_clone(text=text, language="Russian",
                                                  voice_clone_prompt=prompt, **GEN_KWARGS)
            torch.cuda.synchronize()
            dt = time.time() - t
            dur = len(wavs[0]) / sr
            print(f"BENCH {name} run{i} synth={dt:.3f}s audio={dur:.2f}s rtf={dt/dur:.3f} "
                  f"peak_gb={torch.cuda.max_memory_allocated()/1e9:.2f}", flush=True)
            if i == 0 and name == "full":
                sf.write(os.path.join(a.out, "qwen3-bench-full.wav"), wavs[0], sr)
    print("GPU_TOTAL_RESERVED_GB %.2f" % (torch.cuda.max_memory_reserved() / 1e9), flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
