# CosyVoice 3 (Fun-CosyVoice3-0.5B)

- **Kind:** TTS, local (GPU), zero-shot voice cloning
- **Upstream:** https://github.com/FunAudioLLM/CosyVoice, weights: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- **Status in this project:** investigated; **no Russian synthesis result was recorded**. This card documents what was checked in the model card and source code only. Treat Russian quality as unknown/unproven.

## License of code and weights

- Weights: **Apache-2.0** (Hugging Face model card header `License: apache-2.0`, checked 2026-10-22). Commercial use: **allowed** under Apache-2.0 terms. Not checked: licenses of the training data, or any additional terms in the repository.
- Code: **unverified** (the repository LICENSE file was not read at write time).

## Install and run

Installation steps were not recorded in the project notes; follow the upstream README (Python env, download of the 0.5B checkpoint; the repo ships `asset/zero_shot_prompt.wav` as the engine's own reference sample). Inference is zero-shot: you pass a reference clip plus its transcript and the text to speak.

## Hardware used

A desktop with an NVIDIA RTX 3090 Ti (24 GB), Windows 11 (CPU/RAM not recorded). Nothing was run on CPU.

## What the model card and code say about Russian

| Claim / fact | Where checked |
|---|---|
| "Covers 9 common languages (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, **Russian**)" -- Russian is officially listed | HF model card, 2026-10-22 |
| 0.5 B parameters | model card |
| The special-token vocabulary (`CosyVoice3Tokenizer`, `cosyvoice/tokenizer/tokenizer.py`) contains the full English ARPAbet set and the full Chinese pinyin set for forced pronunciation, and **no Russian phoneme tokens** | read in source |
| CosyVoice 3 has **no `<|ru|>`-style language tag**; language is implied by the reference clip's language and by a natural-language instruction (e.g. `You are a helpful assistant. ... <|endofprompt|>`) | read in source |
| An open upstream issue (`FunAudioLLM/CosyVoice` #1790, as recorded in notes) reports Chinese-sounding speech for Russian input; no maintainer reply at the time | reported by a third party; **not reproduced** here |

Practical reading: English and Chinese have an emergency lever (dictate pronunciation by phonemes); Russian has none, so quality depends on how much Russian was in the training data. No stress control for Russian is exposed.

## Planned experiment (not completed in the notes)

Question: is poor cloning quality caused by the engine or by a low-quality reference clip? Design: same engine, same three Russian phrases, vary only the reference: (a) the engine's own studio reference (ceiling), (b) a clip from a compressed voice message, (c) a clip of synthetic speech. Metrics planned: UTMOS naturalness (English-trained, so only for relative comparison), CAM++ speaker similarity (engine's own encoder; not comparable with other encoders' numbers), energy above 4 and 8 kHz, RTF. **No results were written down**, so none are reported here.

Supporting evidence from a different engine in this repo (XTTS): naturalness fell with reference quality while the engine stayed the same -- see `docs/methodology/liveliness-and-brightness.md` for the related reference-selection findings. Measured: Telegram-style voice messages are Opus mono at roughly 29-30 kbit/s, and a clip taken from one had about 7 dB less energy above 4 kHz than a cleaner clip (-23.3 dB vs -15.9 dB relative to whole-file level; above 8 kHz -30.1 dB vs -20.6 dB). Caveat: the cleaner clip was itself synthetic speech, so this comparison alone is not proof.

## Measured numbers

None recorded (no RTF, latency or WER/CER for Russian).

## Known pitfalls

- Russian is "supported" on paper only; verify with your own listening test before adopting.
- UTMOS is trained on English: compare recordings with each other, do not quote an absolute score as a Russian MOS.
- CAM++ similarity numbers are only comparable within one table.

## Verdict

Permissive license and an official Russian claim, but no Russian-specific controls, an unreproduced upstream report of wrong-language output, and no measurements here: unproven for Russian. Run a short listening test before investing.
