# XTTS-v2 (Coqui)

**Kind:** TTS, zero-shot voice cloning from ~6 s, 17 languages incl. Russian, built-in speakers, streaming output, local GPU.
**Upstream:** weights <https://huggingface.co/coqui/XTTS-v2>; maintained code fork
<https://github.com/idiap/coqui-ai-TTS> (PyPI `coqui-tts`; the original `coqui-ai/TTS` is abandoned, Coqui the company closed).
Tested: September 2026, `coqui-tts 0.27.5`, `transformers 4.57.6`, `torch 2.6.0+cu124`.

## License (checked on the official pages)

| Part | License | Source |
|---|---|---|
| **Weights** | **Coqui Public Model License (CPML) - NON-COMMERCIAL** | HF card label "License: coqui-public-model-license"; `LICENSE.txt` in the weights repo |
| Code (idiap fork) | Mozilla Public License 2.0 | `LICENSE.txt` in the GitHub repo |

Verbatim from the weights' `LICENSE.txt`:
> This license allows only non-commercial use of a machine learning model and its outputs.

> Non-commercial purposes include any of the following uses of the model or its output, but only so far as you do not
> receive any direct or indirect payment arising from the use of the model or its output.

> Use of the model to train other models for commercial use is not a non-commercial purpose.

**Commercial use: NO.** The restriction covers the generated audio too, not only the model. The license link
`https://coqui.ai/cpml` no longer resolves at the time of testing (the company shut down).

## Install & run

```bash
uv venv --python 3.12 venv && source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install coqui-tts "transformers<5"                  # pin transformers right away, see pitfalls
export COQUI_TOS_AGREED=1 TTS_HOME=./models             # COQUI_TOS_AGREED=1 = you accept the CPML
python tts/xtts/run_xtts.py --ref voiceA=./refs/voiceA.wav --stream
python tts/xtts/similarity.py --refs-dir ./refs ./out/xtts
python tts/xtts/mos.py ./out/xtts
```
Model `tts_models/multilingual/multi-dataset/xtts_v2` (1.87 GB) downloads on first run. Reference: 6-15 s of clean
speech, no transcript required; use recordings you have the rights to.

## Hardware for the measurements

Windows 11, **NVIDIA RTX 3090 Ti 24 GB**, GPU inference. CPU and RAM were not recorded. Not run on CPU.

## Measured numbers

The scripts print `rtf` as **audio/synth (higher = faster)**; below the usual RTF = synth/audio is derived as 1/x.

| Metric | Value |
|---|---|
| Speed, 5 phrases | audio/synth **1.87-2.09** (derived RTF ~0.48-0.53); 1.8-3.9 s per phrase |
| **Time to first audio** (`inference_stream`, `stream_chunk_size=20`) | **470-531 ms** |
| VRAM | peak allocated 1.90-1.98 GB; `nvidia-smi` delta **+2.1 GB after load, +2.6 GB after synthesis** |
| Load | 16 s with warm disk; 198 s on first run including the 1.87 GB download |
| Round-trip STT (Gemini 2.5 Flash), WER excluding numerals | **6.4%**, 1 of 15 files with a real error (an appended extra word) |
| Speaker similarity to the reference (XTTS's own speaker encoder, cosine) | clone of voice B 0.686 (0.671-0.707); clone of voice A (phone-quality clip) **0.580** (0.518-0.625); built-in speaker as control 0.295 |
| UTMOS (English-trained predictor, relative use only) | built-in speaker **3.84** (worst 3.48); clone B 3.13; clone A 2.70; the references themselves 2.56 (B) and 1.46 (A) |

## Quality notes for Russian

- Text is read faithfully (14 of 15 files word for word; numerals were written as words in the test phrases).
- **Cleans the reference:** output is studio-like even from a codec-compressed phone clip (clone A: UTMOS 2.70 vs 1.46
  for the clip), but loses recognisability (similarity 0.58 vs 0.70 for F5 on the same clip).
- Keeps its own speaking tempo regardless of the reference (F5 copies the reference tempo).
- Built-in speaker is a non-Russian voice, so an accent is likely - **not listened to**, no number covers accent,
  question intonation or pauses.
- Number reading from digits, and 8 kHz telephone-format quality were not tested.
- Output peaks reach 0.0 dBFS on some phrases: normalise loudness before telephony or you get clicks.

## Known pitfalls

1. Fresh `coqui-tts` pulls `transformers 5.x`, which removed `isin_mps_friendly`:
   `ImportError: cannot import name 'isin_mps_friendly' from 'transformers.pytorch_utils'`. Install `"transformers<5"`.
2. Needs a separate venv from F5-TTS (incompatible `transformers` requirements; costs a second copy of torch ~2.5 GB).
3. On Windows the console codepage (cp1251) crashes scripts that print `->`-like symbols: `PYTHONIOENCODING=utf-8`.
4. Long SSH sessions drop while a ~1.3 GB model downloads; use `ServerAliveInterval`.
5. With 21.5 GB of 24 GB taken by an LLM, XTTS (+2.6 GB) does not fit.

## Verdict

Lowest time-to-first-audio of the open cloners tested (0.5 s, real streaming) and the cleanest built-in voice, but the
weights are **non-commercial (CPML, covering outputs)**, cloning is less faithful than F5/ESpeech, and the reference gets
"cleaned". Fine for prototypes and research; not for commercial calls.
