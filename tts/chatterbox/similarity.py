"""Speaker similarity: cosine between speaker embeddings (Chatterbox VoiceEncoder).

The first file is the reference (real recording); every other file is compared
to it. Caveat: this is the same encoder the model conditions on, so the metric
is somewhat favourable to the model.

Usage:
    MODELS_DIR=./models python3 similarity.py ref.wav clone1.wav other.wav ...
"""
import os
import sys

import librosa
import numpy as np
import torch

torch.set_num_threads(4)
from chatterbox.models.voice_encoder import VoiceEncoder  # noqa: E402

MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
ve = VoiceEncoder()
ve.load_state_dict(torch.load(os.path.join(MODELS_DIR, "ve.pt"),
                              map_location="cpu", weights_only=True))
ve.eval()


def emb(path):
    wav, _ = librosa.load(path, sr=16000)
    e = np.asarray(ve.embeds_from_wavs([wav], sample_rate=16000)).mean(axis=0)
    return e / np.linalg.norm(e)


names = sys.argv[1:]
if len(names) < 2:
    sys.exit("usage: similarity.py REF.wav OTHER.wav [OTHER.wav ...]")
E = {n: emb(n) for n in names}
base = names[0]
print(f"reference: {base}\n")
for n in names[1:]:
    print(f"  cos={float(E[base] @ E[n]):.3f}   {n}")
