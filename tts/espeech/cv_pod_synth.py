"""Two questions in one run:
  1) control: same engine, but references of people READING aloud (e.g. Common Voice
     clips) - how much duller is the result than with conversational references?
  2) the ESpeech "podcaster" checkpoint vs the RL-V2 checkpoint on the same references.

Usage:
  python cv_pod_synth.py --live-refs ./refs/live_refs.json --read-refs ./refs/read_refs.json \
         [--rl-ckpt ./models/espeech_tts_rlv2.pt] [--podcaster-ckpt ./models/espeech_tts_podcaster.pt] \
         [--vocab ./models/vocab.txt] [--out ./out/cvpod]

refs JSON format: see common.load_refs. Needs: GPU, `pip install f5-tts ruaccent soundfile`.
A missing checkpoint is skipped. Fixed seed; writes cvpod.json.
"""
import argparse
import gc
import json
import os
import time

import soundfile as sf
import torch

import common
common.patch_audio_io()

SEED = 2002
P = {"p1": ("Здравствуйте! Меня зовут Кирилл, барбершоп «Ромашка». Вам удобно сейчас говорить?",
            "Здравствуйте! Э-э... меня зовут Кирилл, барбершоп «Ромашка». Вам сейчас удобно говорить?"),
     "p2": ("Ваша запись уже готова, я звоню уточнить: приходите сами или перенести на другой день?",
            "Ваша запись, ну, уже готова. Я звоню уточнить - приходите сами, или, может, переносим на другой день?"),
     "p3": ("К сожалению, мастер задерживается примерно на полчаса. Извините за неудобство.",
            "Ну, к сожалению, мастер задерживается... примерно на полчаса. Извините, эм, за неудобство.")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--live-refs", default="./refs/live_refs.json")
    ap.add_argument("--read-refs", default="./refs/read_refs.json")
    ap.add_argument("--rl-ckpt", default=common.DEFAULT_CKPT)
    ap.add_argument("--podcaster-ckpt", default="./models/espeech_tts_podcaster.pt")
    ap.add_argument("--vocab", default=common.DEFAULT_VOCAB)
    ap.add_argument("--out", default="./out/cvpod")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    live, read = common.load_refs(a.live_refs), common.load_refs(a.read_refs)

    from f5_tts.infer.utils_infer import (infer_process, load_model, load_vocoder,
                                          preprocess_ref_audio_text)
    from f5_tts.model import DiT
    from ruaccent import RUAccent
    dev = torch.device("cuda")
    vocoder = load_vocoder()
    vocoder.to(dev)
    acc = RUAccent()
    acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
    accent = common.make_accent(acc)
    res = []

    def run(tag, ckpt, refset):
        if not os.path.exists(ckpt):
            print(f"SKIP {tag}: no file {ckpt}", flush=True)
            return
        m = load_model(DiT, common.MODEL_CFG, ckpt, vocab_file=a.vocab)
        m.to(dev)
        prep = {k: preprocess_ref_audio_text(r["file"], accent(r["text"])) for k, r in refset.items()}
        for pid, (plain, conv) in P.items():
            for vname, text in (("plain", plain), ("conversational", conv)):
                g = accent(text)
                for k in refset:
                    ra, rt = prep[k]
                    common.seed_all(SEED)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    w, sr, _ = infer_process(ra, rt, g, m, vocoder, cross_fade_duration=0.15,
                                             nfe_step=32, speed=1.0)
                    torch.cuda.synchronize()
                    dt = time.time() - t1
                    dur = len(w) / sr
                    sfx = "" if vname == "plain" else "-conv"
                    nm = f"{tag}-{k}-{pid}{sfx}.wav"
                    sf.write(os.path.join(a.out, nm), w, sr)
                    print(f"RESULT {nm} synth={dt:.2f}s audio={dur:.2f}s rtf={dt/dur:.3f}", flush=True)
                    res.append(dict(name=nm, model=tag, ref=k, phrase=pid, variant=vname,
                                    rtf=dt / dur, audio=dur))
            with open(os.path.join(a.out, "cvpod.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
        del m
        gc.collect()
        torch.cuda.empty_cache()
        print(f"MODEL_DONE {tag}", flush=True)

    run("espeech-read", a.rl_ckpt, read)             # 1) control: read-aloud references
    run("podcaster-live", a.podcaster_ckpt, live)    # 2) podcaster checkpoint on live references
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
