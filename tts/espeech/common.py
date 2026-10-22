"""Shared helpers for the ESpeech / F5-TTS scripts in this directory.

Not a script. Imported by espeech_synth.py, live_synth.py, grid_synth.py,
cv_pod_synth.py, zvonok_synth.py and by ../zv_esp.py, ../esp_stream_bench.py.

Conventions (all overridable from the CLI of each script):
  ./models/espeech_tts_rlv2.pt + ./models/vocab.txt   ESpeech weights (env ESPEECH_CKPT, ESPEECH_VOCAB)
  ./refs/refs.json                                    reference voices (see load_refs)
  ./out/                                              output directory
"""
import json
import os

import soundfile as sf
import torch

# Same architecture config as published on the ESpeech model card.
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

DEFAULT_CKPT = os.environ.get("ESPEECH_CKPT", "./models/espeech_tts_rlv2.pt")
DEFAULT_VOCAB = os.environ.get("ESPEECH_VOCAB", "./models/vocab.txt")
DEFAULT_REFS = os.environ.get("RSB_REFS", "./refs/refs.json")
DEFAULT_PHRASES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "qwen3", "phrases.json")


def patch_audio_io():
    """Replace torchaudio.load/save with soundfile.

    torchaudio >= 2.9 delegates audio I/O to torchcodec, which needs FFmpeg shared
    libraries. On machines without them `import torchcodec` succeeds but the first
    real read fails ("Could not load libtorchcodec"). soundfile has no such dependency.
    """
    import torchaudio

    def _load(path, *a, **k):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    def _save(path, tensor, sr, *a, **k):
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T
        sf.write(str(path), arr, sr)

    torchaudio.load = _load
    torchaudio.save = _save


def load_refs(path=DEFAULT_REFS):
    """Load a reference-voice table.

    JSON format: {"voiceA": {"file": "voiceA.wav", "text": "exact transcript"}, ...}
    "file" is resolved relative to the directory of the JSON file. Put your own
    reference recordings (clean speech, 8-12 s for ESpeech, audio you have the
    rights to use) into ./refs/.
    Returns {name: {"file": absolute_path, "text": transcript}}.
    """
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        refs = json.load(f)
    for r in refs.values():
        r["file"] = os.path.join(base, r["file"])
    return refs


def load_espeech(ckpt=DEFAULT_CKPT, vocab=DEFAULT_VOCAB, device="cuda"):
    """Return (model, vocoder, accentizer) with weights on `device`."""
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT
    from ruaccent import RUAccent

    model = load_model(DiT, MODEL_CFG, ckpt, vocab_file=vocab)
    vocoder = load_vocoder()
    acc = RUAccent()
    acc.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
    dev = torch.device(device)
    model.to(dev)
    vocoder.to(dev)
    return model, vocoder, acc


def make_accent(acc):
    """Text -> text with '+' stress marks; text that already has '+' is left alone."""
    def accent(t):
        if "+" in t:
            return t
        try:
            return acc.process_all(t)
        except Exception as e:  # keep going on RUAccent failures
            print("ACCENT_FAIL", repr(t)[:60], e, flush=True)
            return t
    return accent


def seed_all(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
