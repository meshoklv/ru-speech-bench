"""ESpeech with conversational reference clips: does a spontaneous-speech reference
give a livelier voice than a read-aloud one, and do filler words in the text help?

Usage:
  python live_synth.py --refs ./refs/live_refs.json [--out ./out/live]

refs JSON format: see common.load_refs. Needs: GPU, `pip install f5-tts ruaccent soundfile`,
ESpeech weights (see espeech_synth.py). The seed is fixed, so the "plain" and
"conversational" variants of a phrase differ only in the text.
"""
import argparse
import json
import os
import time

import soundfile as sf
import torch

import common
common.patch_audio_io()

SEED = 2002
PHRASES = {
    "p1": {
        "plain": "Здравствуйте! Меня зовут Кирилл, барбершоп «Ромашка». Вам удобно сейчас говорить?",
        "conversational": "Здравствуйте! Э-э... меня зовут Кирилл, барбершоп «Ромашка». Вам сейчас удобно говорить?",
    },
    "p2": {
        "plain": "Ваша запись уже готова, я звоню уточнить: приходите сами или перенести на другой день?",
        "conversational": "Ваша запись, ну, уже готова. Я звоню уточнить - приходите сами, или, может, переносим на другой день?",
    },
    "p3": {
        "plain": "К сожалению, мастер задерживается примерно на полчаса. Извините за неудобство.",
        "conversational": "Ну, к сожалению, мастер задерживается... примерно на полчаса. Извините, эм, за неудобство.",
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--refs", default="./refs/live_refs.json")
    ap.add_argument("--out", default="./out/live")
    ap.add_argument("--ckpt", default=common.DEFAULT_CKPT)
    ap.add_argument("--vocab", default=common.DEFAULT_VOCAB)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    refs = common.load_refs(a.refs)

    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    model, vocoder, acc = common.load_espeech(a.ckpt, a.vocab)
    accent = common.make_accent(acc)

    prep = {}
    for k, r in refs.items():
        prep[k] = preprocess_ref_audio_text(r["file"], accent(r["text"]))
        print(f"REF_READY {k}", flush=True)

    results = []
    for pid, variants in PHRASES.items():
        for vname, text in variants.items():
            gtext = accent(text)
            print(f"TEXT {pid}/{vname}: {gtext}", flush=True)
            for k in refs:
                ref_proc, rtext_proc = prep[k]
                common.seed_all(SEED)
                torch.cuda.synchronize()
                t1 = time.time()
                wave, sr, _ = infer_process(ref_proc, rtext_proc, gtext, model, vocoder,
                                            cross_fade_duration=0.15, nfe_step=32, speed=1.0)
                torch.cuda.synchronize()
                dt = time.time() - t1
                dur = len(wave) / sr
                suffix = "" if vname == "plain" else "-conv"
                name = f"espeech-live-{k}-{pid}{suffix}.wav"
                sf.write(os.path.join(a.out, name), wave, sr)
                print(f"RESULT {name} synth={dt:.2f}s audio={dur:.2f}s rtf={dt/dur:.3f}", flush=True)
                results.append(dict(name=name, ref=k, phrase=pid, variant=vname, synth=dt,
                                    audio=dur, rtf=dt / dur, text=text, accented=gtext))
            with open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
