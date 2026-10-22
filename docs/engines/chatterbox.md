# Chatterbox Multilingual (TTS)

**Kind:** TTS with zero-shot voice cloning, local (GPU strongly recommended) |
**Upstream:** https://github.com/resemble-ai/chatterbox , weights https://huggingface.co/ResembleAI/chatterbox

## What it is

Resemble AI's open TTS (0.5B parameters for the multilingual V3 release per the model card). The
multilingual model supports 23 languages including Russian (`language_id="ru"`; verified in
`SUPPORTED_LANGUAGES` of `chatterbox/mtl_tts.py`). It clones a voice from a short reference clip.

## License of code and weights (checked 2026-10)

| Part | License | Commercial use | Source |
|---|---|---|---|
| Code | **MIT** (Copyright 2025 Resemble AI) | yes | github.com/resemble-ai/chatterbox, `LICENSE` |
| Weights | **MIT** (HF model-card metadata `license: mit`; no separate LICENSE file in the weights repo) | yes | huggingface.co/ResembleAI/chatterbox |

Caveats that are not license restrictions but matter in production:

* **Every generated file carries Resemble AI's Perth watermark** (card: "imperceptible neural watermarks that
  survive MP3 compression, audio editing, and common manipulations"). Output is technically identifiable as
  Chatterbox synthesis. The `resemble-perth` package is a dependency.
* The card's disclaimer: "Don't use this model to do bad things." It has no legal force beyond MIT.
* Cloning a real person's voice is governed by privacy / personality rights, not by the model license.

## Install and run

```bash
uv pip install chatterbox-tts                # 0.1.7 in our run
```

* **Pitfall: it replaces the CPU torch wheel with the CUDA one** (`torch==2.6.0`, 13 `nvidia-*` packages +
  `triton`): free disk dropped 17 GB -> 11 GB. On a CPU-only box repair it afterwards (installing CPU torch
  first does not help, it gets replaced):

  ```bash
  uv pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
      --reinstall-package torch --reinstall-package torchaudio
  uv pip uninstall triton nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
      nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
      nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
      nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
  ```

  Environment 5.9 GB -> 1.6 GB.
* **Weights download:** `huggingface_hub.snapshot_download` stalled on the big files (0 bytes in 6 min, silent
  exit); plain `curl -C -` worked at ~4 MB/s. Scripts: `tts/chatterbox/dl.py`, `tts/chatterbox/fetch.sh`.
* Disk: ~3.2 GB of weights (`s3gen.pt` 1057 MB, `t3_mtl23ls_v2.safetensors` 2144 MB, `ve.pt` 5.7 MB, tokenizer
  JSONs) + 1.6 GB environment + a 34 MB spaCy model downloaded on first run = ~4.9 GB.
* **Version mismatch:** the card describes Multilingual V3 (`t3_model="v3"`), but `chatterbox-tts==0.1.7`
  from PyPI has no such parameter and `from_local()` hard-codes `t3_mtl23ls_v2.safetensors`. **All numbers
  below are V2.** V3 needs newer code (install from git); not measured.
* Run: `MODELS_DIR=./models python3 tts/chatterbox/synth.py`; cloning:
  `REF=./refs/your_ref.wav TAG=cloned python3 tts/chatterbox/synth.py`. Use a reference of >= 10 s of clean
  speech from a speaker whose permission you have.

## Hardware used for the measurements

CPU only, no GPU: Intel Xeon E5-2695 v2 @ 2.4 GHz (8 cores, flags `sse4_2 avx`; **no AVX2, no FMA**),
14 GB RAM available. Part of the runs was done under load average 21 on 8 cores, part on an idle machine
(load 1.05). Speed numbers do **not** represent a modern CPU or any GPU.

## Measured numbers

Source: console output of `tts/chatterbox/synth*.py` (September 2026); no JSON file was saved.
RTF = synthesis seconds / audio seconds (lower is faster).

| Case | Synthesis | Audio | RTF | Load |
|---|---|---|---|---|
| built-in voice, phrase 1 | 320.9 s | 4.2 s | 76.41 | loaded machine |
| built-in voice, phrase 2 | 140.6 s | 5.5 s | 25.65 | loaded machine |
| cloned, phrase 1 | 159.5 s | 4.9 s | 32.68 | idle |
| cloned, phrase 2 | 222.4 s | 7.3 s | 30.55 | idle |
| digits sentence | 130.3 s | 5.8 s | 22.47 | idle |
| spelled-out sentence | 143.0 s | 5.9 s | 24.32 | idle |

* Model load 62.3 s; **peak RSS 6793 MB** for one process.
* Phrase 1 of `synth.py` was reworded for the public release (company name removed), so a re-run
  will not reproduce the stored durations exactly.
* Idle machine still gives RTF 22-24: the bottleneck is the CPU (no AVX2/FMA), not the neighbours.
* Speaker similarity (cosine of Chatterbox `VoiceEncoder` embeddings, reference = a real 12.3 s clip,
  98 % speech): another clip of the same real voice 0.887 (ceiling); clone 0.821 and 0.802; unrelated
  built-in voice 0.740 and 0.700. The clone covers roughly 60 % of the gap to the real voice. The metric uses
  the model's own encoder, so it flatters the model.
* Intelligibility check (synthesize -> external ASR (Gemini 2.5 Flash) -> compare with the input text):
  text without digits came back verbatim. See `docs/methodology/tts-roundtrip-checks.md`.

## Quality notes for Russian

* **Digits are read as garbage.** "заказ номер 4521 на сумму 15 750 рублей" came back from ASR as nonsense
  ("на меру", "комбастян"), and the log showed `Detected 2x repetition of token ...` (generation aborted).
  **Spelled-out numbers are pronounced flawlessly** (ASR returned "No 4521" and "15 750 ₽"). Always
  normalize numbers to words before synthesis.
* **Stress marks are not applied.** The tokenizer's Russian branch needs the package `russian_text_stresser`;
  the log says "russian_text_stresser not available - Russian stress labeling skipped" and the package was
  not found on PyPI (404 on `russian-text-stresser`). Homographs ("зАмок / замОк") will be wrong.
* Cloning works on Russian but gives "similar", not "same person" (see cosine above).

## Known pitfalls

* Dependency pins override your torch build; budget disk (~5 GB) and repair the CPU wheel if needed.
* Hugging Face downloads may stall; use `curl -C -`.
* PyPI gives V2, the card advertises V3.
* Watermark on all output.

## Verdict

Good Russian pronunciation on words and working cloning, permissive MIT license, but unusable on CPU
(RTF 22-76, 6.8 GB RAM), needs number normalization and has no Russian stress. Worth re-measuring on a GPU
and with V3; GPU speed and V3 quality are **not measured** here.
