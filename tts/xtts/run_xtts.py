"""XTTS-v2 (fork idiap/coqui-ai-TTS, package `coqui-tts`): synthesize 5 phrases + benchmark.

Usage:
  COQUI_TOS_AGREED=1 python run_xtts.py [--ref voiceA=./refs/voiceA.wav ...] [--stream] \
        [--out ./out/xtts] [--models ./models]

Always runs the built-in speaker "Ana Florence"; every --ref NAME=PATH (repeatable) adds a
voice clone from that clip (6-15 s of clean speech; no transcript needed; use audio you
have the rights to). Measures synth time, "rtf", peak VRAM and, with --stream, the latency
to the first chunk of inference_stream.
NOTE: "rtf" printed here is audio_time / synth_time (HIGHER = faster, 1.0 = real time) -
the inverse of the usual RTF.
Needs: GPU, `pip install coqui-tts "transformers<5"`. The XTTS-v2 weights are under the
non-commercial Coqui Public Model License: setting COQUI_TOS_AGREED=1 means YOU accept it.
"""
import argparse
import json
import os
import time

os.environ.setdefault("TTS_HOME", "./models")

import torch
import torchaudio

from phrases import PHRASES

SR = 24000


def vram_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def load_model():
    """Low-level Xtts - needed for inference_stream (first-chunk latency)."""
    from TTS.utils.manage import ModelManager
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    name = "tts_models/multilingual/multi-dataset/xtts_v2"
    mm = ModelManager(output_prefix=os.environ["TTS_HOME"], progress_bar=True)
    model_path, _, _ = mm.download_model(name)
    print(f"[model] {model_path}", flush=True)

    cfg = XttsConfig()
    cfg.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=model_path, eval=True, use_deepspeed=False)
    model.cuda()
    return model, model_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--stream", action="store_true", help="also measure latency to the first chunk")
    ap.add_argument("--ref", action="append", default=[], metavar="NAME=PATH",
                    help="voice-clone reference (repeatable)")
    ap.add_argument("--out", default="./out/xtts")
    ap.add_argument("--models", default=None, help="TTS_HOME (default ./models)")
    a = ap.parse_args()
    if a.models:
        os.environ["TTS_HOME"] = a.models

    os.makedirs(a.out, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA is not available"
    print(f"[gpu] {torch.cuda.get_device_name(0)}  torch={torch.__version__}", flush=True)

    t0 = time.time()
    model, model_path = load_model()
    load_s = time.time() - t0
    print(f"[load] model up in {load_s:.1f} s, VRAM={vram_mb():.0f} MB", flush=True)

    voices = {"builtin": None}
    for item in a.ref:
        name, path = item.split("=", 1)
        voices[f"clone-{name}"] = path

    results = {"load_s": round(load_s, 2), "vram_after_load_mb": round(vram_mb()),
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "runs": []}

    for vname, ref in voices.items():
        t = time.time()
        if ref is None:
            # conditioning latents of a built-in speaker from speakers_xtts.pth
            spk = torch.load(os.path.join(model_path, "speakers_xtts.pth"), map_location="cpu")
            key = "Ana Florence" if "Ana Florence" in spk else sorted(spk)[0]
            gpt_cond = spk[key]["gpt_cond_latent"].cuda()
            spk_emb = spk[key]["speaker_embedding"].cuda()
            print(f"[voice] {vname}: built-in '{key}'", flush=True)
        else:
            gpt_cond, spk_emb = model.get_conditioning_latents(
                audio_path=[ref], gpt_cond_len=30, max_ref_length=60)
            print(f"[voice] {vname}: clone from {os.path.basename(ref)}", flush=True)
        cond_s = time.time() - t

        for key_, text in PHRASES:
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            out = model.inference(text, "ru", gpt_cond, spk_emb,
                                  temperature=0.7, enable_text_splitting=True)
            dt = time.time() - t
            wav = torch.tensor(out["wav"]).unsqueeze(0)
            dur = wav.shape[1] / SR
            path = os.path.join(a.out, f"xtts-{vname}-{key_}.wav")
            torchaudio.save(path, wav, SR)
            results["runs"].append({"voice": vname, "phrase": key_, "synth_s": round(dt, 2),
                                    "audio_s": round(dur, 2), "rtf": round(dur / dt, 2),
                                    "peak_vram_mb": round(vram_mb()), "file": os.path.basename(path)})
            print(f"  {key_:12s} synth {dt:5.2f} s / audio {dur:5.2f} s / audio-per-synth {dur/dt:5.2f} / "
                  f"VRAM {vram_mb():.0f} MB", flush=True)

        if a.stream:
            text = PHRASES[1][1]  # the longest phrase
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            first, chunks = None, []
            for ch in model.inference_stream(text, "ru", gpt_cond, spk_emb, stream_chunk_size=20):
                if first is None:
                    first = time.time() - t
                chunks.append(ch.cpu())
            total = time.time() - t
            wav = torch.cat(chunks).unsqueeze(0)
            path = os.path.join(a.out, f"xtts-{vname}-stream.wav")
            torchaudio.save(path, wav, SR)
            results["runs"].append({"voice": vname, "phrase": "stream(zakaz)",
                                    "first_chunk_s": round(first, 3), "synth_s": round(total, 2),
                                    "audio_s": round(wav.shape[1] / SR, 2),
                                    "rtf": round(wav.shape[1] / SR / total, 2),
                                    "peak_vram_mb": round(vram_mb()), "file": os.path.basename(path)})
            print(f"  stream       first chunk {first*1000:.0f} ms / total {total:.2f} s / "
                  f"VRAM {vram_mb():.0f} MB", flush=True)

        results.setdefault("cond_s", {})[vname] = round(cond_s, 2)

    with open(os.path.join(a.out, "bench_xtts.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[ok] results in bench_xtts.json", flush=True)


if __name__ == "__main__":
    main()
