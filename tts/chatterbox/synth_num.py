"""Chatterbox Russian: numbers as digits vs. spelled out.

Synthesizes the same sentence twice (digits, then words) so the outputs can be
round-tripped through an ASR and compared. Digits produced garbage in our runs.

Usage:
    MODELS_DIR=./models OUT_DIR=./out python3 synth_num.py
"""
import os
import time

import torch

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
import torchaudio as ta  # noqa: E402
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
OUT = os.environ.get("OUT_DIR", "./out")
os.makedirs(OUT, exist_ok=True)

m = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device="cpu")
PH = [
    ("num-digits", "Ваш заказ номер 4521 на сумму 15 750 рублей готов к выдаче."),
    ("num-words", "Ваш заказ номер сорок пять двадцать один на сумму "
                  "пятнадцать тысяч семьсот пятьдесят рублей готов к выдаче."),
]
for name, text in PH:
    t0 = time.time()
    wav = m.generate(text, language_id="ru")
    el = time.time() - t0
    dur = wav.shape[-1] / m.sr
    ta.save(f"{OUT}/{name}.wav", wav, m.sr)
    print(f"[{name}] synth {el:.1f}s audio {dur:.1f}s RTF={el / dur:.2f}")
    print(f"      input: {text}")
