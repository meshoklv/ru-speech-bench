"""Chatterbox Multilingual TTS: Russian synthesis, built-in or cloned voice.

Measures model load time, synthesis time, RTF (synthesis_s / audio_s; lower is
faster) and peak RSS.

Usage:
    pip install chatterbox-tts        # see docs/engines/chatterbox.md for pitfalls
    MODELS_DIR=./models OUT_DIR=./out python3 synth.py                  # built-in voice
    REF=./refs/my_ref.wav TAG=cloned python3 synth.py                   # voice cloning
Env: THREADS (default 6), MODELS_DIR, OUT_DIR, REF (>=10 s of clean speech you
have the rights to use), TAG (output file prefix). CPU only as written.
"""
import os
import resource
import time

import torch

torch.set_num_threads(int(os.environ.get("THREADS", "6")))
import torchaudio as ta  # noqa: E402
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS  # noqa: E402

MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
OUT = os.environ.get("OUT_DIR", "./out")
os.makedirs(OUT, exist_ok=True)

print("'ru' in supported languages:", "ru" in SUPPORTED_LANGUAGES, "->", SUPPORTED_LANGUAGES.get("ru"))
t0 = time.time()
m = ChatterboxMultilingualTTS.from_local(MODELS_DIR, device="cpu")
print(f"model load: {time.time() - t0:.1f}s  sr={m.sr}  "
      f"RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")

PHRASES = [
    ("p1", "Здравствуйте! Я звоню из клиники по поводу вашей записи."),
    ("p2", "Ваш заказ номер 4521 на сумму 15 750 рублей готов к выдаче."),
]
ref = os.environ.get("REF")            # None = built-in voice
tag = os.environ.get("TAG", "builtin")
for name, text in PHRASES:
    t0 = time.time()
    kw = dict(language_id="ru")
    if ref:
        kw["audio_prompt_path"] = ref
    wav = m.generate(text, **kw)
    el = time.time() - t0
    dur = wav.shape[-1] / m.sr
    path = f"{OUT}/{tag}-{name}.wav"
    ta.save(path, wav, m.sr)
    print(f"[{tag}/{name}] synth {el:.1f}s  audio {dur:.1f}s  RTF={el / dur:.2f}  -> {path}")
    print(f"           text: {text}")
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
