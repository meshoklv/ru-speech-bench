"""Shared helpers for the Qwen3-TTS scripts. Not a script; imported by the others
(and by ../zv_qwen.py, ../k3fix.py).

Conventions (overridable from each script's CLI):
  ./refs/refs.json   reference voices (see load_refs)
  ./out/             output directory
Env: QWEN_MODEL (default Qwen/Qwen3-TTS-12Hz-1.7B-Base), QWEN_ATTN (default sdpa).
"""
import json
import os

import torch

MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
ATTN = os.environ.get("QWEN_ATTN", "sdpa")
DEFAULT_REFS = os.environ.get("RSB_REFS", "./refs/refs.json")
HERE = os.path.dirname(os.path.abspath(__file__))

# Sampling parameters used for every measurement in the docs.
GEN_KWARGS = dict(max_new_tokens=2048, do_sample=True, top_k=50, top_p=1.0,
                  temperature=0.9, repetition_penalty=1.05,
                  subtalker_dosample=True, subtalker_top_k=50,
                  subtalker_top_p=1.0, subtalker_temperature=0.9)


def load_refs(path=DEFAULT_REFS):
    """JSON: {"voiceA": {"file": "voiceA.wav", "text": "exact transcript"}, ...}.
    "file" is relative to the JSON's directory. Put your own reference recordings
    (clean speech you have the rights to use; end the clip on a finished sentence,
    otherwise the model tends to 'finish' the reference's last words) into ./refs/.
    """
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        refs = json.load(f)
    for r in refs.values():
        r["file"] = os.path.join(base, r["file"])
    return refs


def load_model(device="cuda:0"):
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=device,
                                         dtype=torch.bfloat16, attn_implementation=ATTN)


def make_prompts(model, refs):
    """One clone prompt per voice, reused for all lines (keeps the voice stable)."""
    return {n: model.create_voice_clone_prompt(ref_audio=r["file"], ref_text=r["text"],
                                               x_vector_only_mode=False)
            for n, r in refs.items()}


def seed_all(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
