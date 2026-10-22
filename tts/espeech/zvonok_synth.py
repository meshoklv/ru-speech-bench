"""Synthesize a whole phone-call scene (lines k1..k6) with ESpeech for several voices,
in two text variants: plain, and with conversational filler words.
The caller's name and verb gender are adapted to the gender of each voice.

Usage:
  python zvonok_synth.py --voices ./refs/zvonok_voices.json [--out ./out/zvonok_live]

voices JSON: [{"tag": "v1", "file": "v1.wav", "text": "exact transcript", "pol": "m" | "f"}, ...]
("file" is relative to the JSON's directory). Needs: GPU, `pip install f5-tts ruaccent soundfile`,
ESpeech weights (see espeech_synth.py). Fixed seed; writes zvonok.json.
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


def scene(pol):
    name = "Анна" if pol == "f" else "Кирилл"
    booked = "записала" if pol == "f" else "записал"
    Booked = booked.capitalize()
    plain = {
        "k1": f"Здравствуйте! Меня зовут {name}, барбершоп «Ромашка». Вам удобно сейчас говорить?",
        "k2": "Я по поводу вашей записи на стрижку. Мастер уже ждёт вас в субботу.",
        "k3": "Записать можно с завтрашнего дня, салон работает с девяти утра до восьми вечера. Выбираем время или подумаете?",
        "k4": "Стрижка с бородой стоит три тысячи двести рублей. Могу записать на четверг, в первой половине дня.",
        "k5": f"{Booked} на четверг, с десяти до двенадцати. За час мы вам напомним. Что-нибудь ещё подсказать?",
        "k6": "Тогда хорошего дня, до свидания!"}
    conv = {
        "k1": f"Здравствуйте! Э-э... меня зовут {name}, барбершоп «Ромашка». Вам сейчас удобно говорить?",
        "k2": "Я, значит, по поводу вашей записи на стрижку. Мастер уже ждёт вас в субботу.",
        "k3": "Записать можно с завтрашнего дня, салон работает с девяти утра до восьми вечера. Ну что, выбираем время... или подумаете?",
        "k4": "Стрижка с бородой стоит три тысячи двести рублей. Могу, вот, записать на четверг, в первой половине дня.",
        "k5": f"Так, {booked} на четверг, с десяти до двенадцати. За час мы вам напомним. Что-нибудь ещё подсказать?",
        "k6": "Ну тогда хорошего дня, до свидания!"}
    return plain, conv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--voices", default="./refs/zvonok_voices.json")
    ap.add_argument("--out", default="./out/zvonok_live")
    ap.add_argument("--ckpt", default=common.DEFAULT_CKPT)
    ap.add_argument("--vocab", default=common.DEFAULT_VOCAB)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    base = os.path.dirname(os.path.abspath(a.voices))
    with open(a.voices, encoding="utf-8") as f:
        voices = json.load(f)

    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    model, vocoder, acc = common.load_espeech(a.ckpt, a.vocab)
    accent = common.make_accent(acc)

    res = []
    for v in voices:
        ra, rt = preprocess_ref_audio_text(os.path.join(base, v["file"]), accent(v["text"]))
        plain, conv = scene(v["pol"])
        for vname, texts in (("plain", plain), ("conversational", conv)):
            for kid, text in texts.items():
                g = accent(text)
                common.seed_all(SEED)
                torch.cuda.synchronize()
                t1 = time.time()
                w, sr, _ = infer_process(ra, rt, g, model, vocoder, cross_fade_duration=0.15,
                                         nfe_step=32, speed=1.0)
                torch.cuda.synchronize()
                dt = time.time() - t1
                dur = len(w) / sr
                sfx = "" if vname == "plain" else "-conv"
                nm = f"zvonok-{v['tag']}-{kid}{sfx}.wav"
                sf.write(os.path.join(a.out, nm), w, sr)
                print(f"RESULT {nm} synth={dt:.2f}s audio={dur:.2f}s rtf={dt/dur:.3f}", flush=True)
                res.append(dict(name=nm, voice=v["tag"], line=kid, variant=vname, text=text,
                                audio=dur, rtf=dt / dur))
        with open(os.path.join(a.out, "zvonok.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"VOICE_DONE {v['tag']}", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
