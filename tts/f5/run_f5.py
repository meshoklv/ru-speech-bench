"""F5-TTS with the Russian fine-tune (Misha24-10/F5-TTS_RUSSIAN): synthesis + speed/VRAM benchmark.

Usage:
  python run_f5.py [--ckpt v2|v1base] [--accent] [--speed 1.0] \
        [--refs ./refs/refs.json] [--out ./out/f5]

refs.json: {"voiceA": {"file": "voiceA.wav", "text": "exact transcript"}, ...}  ("file" relative
to the JSON's directory; put your own dense, pause-free 8-12 s clips there).
Checkpoints:
  v1base  F5TTS_v1_Base/model_240000_inference.safetensors   stress marks optional
  v2      F5TTS_v1_Base_v2/model_last_inference.safetensors  trained on 100% stress-annotated text
          -> use --accent so the text is fed with '+' marks
The vocab file lives only in F5TTS_v1_Base/vocab.txt and is shared by all checkpoints;
without it (or with the default English one) the output is garbage.
F5 synthesizes a phrase as a whole (flow matching); there is no streaming output.
Needs: GPU, `pip install f5-tts soundfile`. NOTE: "rtf" printed here is audio_time / synth_time
(HIGHER = faster, 1.0 = real time) - the inverse of the usual RTF.
"""
import argparse
import json
import os
import time

import torch
from huggingface_hub import hf_hub_download

from phrases_f5 import PHRASES

REPO = "Misha24-10/F5-TTS_RUSSIAN"
CKPTS = {
    "v1base": "F5TTS_v1_Base/model_240000_inference.safetensors",
    "v2": "F5TTS_v1_Base_v2/model_last_inference.safetensors",
}


def vram_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", default="v2", choices=list(CKPTS))
    ap.add_argument("--accent", action="store_true", help="feed text with stress marks")
    ap.add_argument("--speed", type=float, default=1.0, help="speech rate (1.0 = as in the reference)")
    ap.add_argument("--refs", default=os.environ.get("RSB_REFS", "./refs/refs.json"))
    ap.add_argument("--out", default="./out/f5")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA is not available"
    print(f"[gpu] {torch.cuda.get_device_name(0)}  torch={torch.__version__}", flush=True)

    base = os.path.dirname(os.path.abspath(a.refs))
    with open(a.refs, encoding="utf-8") as f:
        refs = json.load(f)

    t0 = time.time()
    ckpt = hf_hub_download(REPO, CKPTS[a.ckpt])
    vocab = hf_hub_download(REPO, "F5TTS_v1_Base/vocab.txt")
    print(f"[dl] checkpoint {CKPTS[a.ckpt]}\n[dl] vocab {vocab}", flush=True)

    from f5_tts.api import F5TTS
    f5 = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device="cuda")
    load_s = time.time() - t0
    print(f"[load] model up in {load_s:.1f} s, VRAM={vram_mb():.0f} MB", flush=True)

    tag = f"{a.ckpt}{'-akc' if a.accent else ''}{'' if a.speed == 1.0 else f'-sp{a.speed:g}'}"
    results = {"ckpt": CKPTS[a.ckpt], "accent": a.accent, "speed": a.speed, "load_s": round(load_s, 2),
               "vram_after_load_mb": round(vram_mb()), "gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "runs": []}

    for vname, r in refs.items():
        ref = os.path.join(base, r["file"])
        print(f"[voice] {vname}: {r['file']}", flush=True)
        for key, plain, accented in PHRASES:
            text = accented if a.accent else plain
            path = os.path.join(a.out, f"f5-{tag}-{vname}-{key}.wav")
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            wav, sr, _ = f5.infer(ref_file=ref, ref_text=r["text"], gen_text=text,
                                  file_wave=path, remove_silence=False, seed=1234,
                                  speed=a.speed)
            dt = time.time() - t
            dur = len(wav) / sr
            results["runs"].append({"voice": vname, "phrase": key, "synth_s": round(dt, 2),
                                    "audio_s": round(dur, 2), "rtf": round(dur / dt, 2),
                                    "peak_vram_mb": round(vram_mb()), "file": os.path.basename(path)})
            print(f"  {key:12s} synth {dt:5.2f} s / audio {dur:5.2f} s / audio-per-synth {dur/dt:5.2f} / "
                  f"VRAM {vram_mb():.0f} MB", flush=True)

    with open(os.path.join(a.out, f"bench_f5_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[ok] results in bench_f5_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
