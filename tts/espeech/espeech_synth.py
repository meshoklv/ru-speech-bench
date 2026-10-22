"""Synthesize the test phrases with ESpeech-TTS-1 RL-V2 (F5-TTS architecture + RUAccent
stress marks) and record RTF / peak VRAM.

Usage:
  python espeech_synth.py [--refs ./refs/refs.json] [--phrases ../qwen3/phrases.json] [--out ./out/espeech]
  # weights: ./models/espeech_tts_rlv2.pt + ./models/vocab.txt (see dl-espeech.ps1), or
  # set ESPEECH_CKPT / ESPEECH_VOCAB; if missing they are downloaded from the Hugging Face hub.

Needs: GPU, `pip install f5-tts ruaccent soundfile`. The model card limits the reference
clip to 12 s. The first voice in refs.json runs all phrases, other voices phrase "1" only.
RTF = synth_time / audio_time (lower is faster).
"""
import argparse
import json
import os
import time

import soundfile as sf
import torch

import common
common.patch_audio_io()

MODEL_REPO = "ESpeech/ESpeech-TTS-1_RL-V2"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--refs", default=common.DEFAULT_REFS)
    ap.add_argument("--phrases", default=common.DEFAULT_PHRASES)
    ap.add_argument("--out", default="./out/espeech")
    ap.add_argument("--ckpt", default=common.DEFAULT_CKPT)
    ap.add_argument("--vocab", default=common.DEFAULT_VOCAB)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(a.phrases, encoding="utf-8") as f:
        phrases = json.load(f)
    refs = common.load_refs(a.refs)

    ckpt, vocab = a.ckpt, a.vocab
    if not (os.path.exists(ckpt) and os.path.exists(vocab)):
        from huggingface_hub import hf_hub_download
        print("downloading weights from the hub...", flush=True)
        ckpt = hf_hub_download(repo_id=MODEL_REPO, filename="espeech_tts_rlv2.pt")
        vocab = hf_hub_download(repo_id=MODEL_REPO, filename="vocab.txt")
    print("ckpt", ckpt, flush=True)

    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    t0 = time.time()
    model, vocoder, acc = common.load_espeech(ckpt, vocab)
    print(f"LOAD_ALL {time.time()-t0:.2f}s", flush=True)
    print(f"VRAM_AFTER_LOAD_GB {torch.cuda.memory_allocated()/1e9:.2f}", flush=True)
    accent = common.make_accent(acc)

    results = []
    for i, (refname, ref) in enumerate(refs.items()):
        ref_proc, rtext_proc = preprocess_ref_audio_text(ref["file"], accent(ref["text"]))
        todo = list(phrases.items()) if i == 0 else [("1", phrases["1"])]
        for pid, text in todo:
            gtext = accent(text)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t1 = time.time()
            wave, sr, _ = infer_process(ref_proc, rtext_proc, gtext, model, vocoder,
                                        cross_fade_duration=0.15, nfe_step=32, speed=1.0)
            torch.cuda.synchronize()
            dt = time.time() - t1
            dur = len(wave) / sr
            peak = torch.cuda.max_memory_allocated() / 1e9
            name = f"espeech-{refname}-phrase{pid}.wav"
            sf.write(os.path.join(a.out, name), wave, sr)
            print(f"RESULT {name} synth={dt:.2f}s audio={dur:.2f}s rtf={dt/dur:.3f} "
                  f"sr={sr} peak_alloc_gb={peak:.2f}", flush=True)
            print(f"  ACCENTED: {gtext}", flush=True)
            results.append(dict(name=name, synth=dt, audio=dur, rtf=dt / dur, sr=sr,
                                peak=peak, accented=gtext))

    with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
