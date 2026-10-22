# Piper (TTS)

**Kind:** TTS, local, CPU-only (ONNX/VITS) | **Upstream:** https://github.com/OHF-Voice/piper1-gpl
(continuation of the archived https://github.com/rhasspy/piper) | **Voices:** https://huggingface.co/rhasspy/piper-voices

## What it is

A small, fast, fully offline neural TTS. A VITS model per voice (~63 MB each, `medium` quality, 22.05 kHz,
single speaker) driven by an espeak-ng phonemizer. Four Russian `medium` voices exist: `denis`, `dmitri`,
`irina`, `ruslan`. No GPU needed.

## License of code and weights (checked 2026-10, upstream pages)

**The runtime and every voice are licensed separately. Read both.**

| Part | License | Commercial use | Source |
|---|---|---|---|
| `piper-tts` runtime (`OHF-Voice/piper1-gpl`) | **GPL-3.0** (repo license page) | yes, but copyleft: linking/distributing it makes your work GPL-3.0 | github.com/OHF-Voice/piper1-gpl |
| Original `rhasspy/piper` (archived 2025-10-06) | MIT (`LICENSE.md`) | yes | github.com/rhasspy/piper |
| `rhasspy/piper-voices` repo metadata | `mit` tag, but per-voice terms live in each `MODEL_CARD` | see below | huggingface.co/rhasspy/piper-voices |

Piper's own voice docs say: "The `MODEL_CARD` file for each voice contains important licensing information"
and "Some voices may have restrictive licenses, however, so please review them carefully!"

Per-voice terms, read from each `ru/ru_RU/<voice>/medium/MODEL_CARD`:

| Voice | Training data (per MODEL_CARD) | Dataset license | Commercial use |
|---|---|---|---|
| `denis` | OHF-Voice/voice-datasets | **CC0** | unverified (see base-model caveat) |
| `dmitri` | OHF-Voice/voice-datasets | **CC0** | unverified (see base-model caveat) |
| `irina` | github.com/RHVoice/RHVoice | **"Unknown"** (stated on the card) | unverified |
| `ruslan` | Ruslan corpus (ruslan-corpus.github.io) | **CC BY-NC-SA 4.0** | **NO (non-commercial)** |

Notes and open questions:

* All four cards say "Finetuned from U.S. English lessac voice (medium quality)". The `lessac` voice is trained
  on the Blizzard 2013 Lessac data, whose license page states that "Research Purposes" excludes any commercial
  purpose. It does not explicitly address models trained on that data. Whether fine-tuned checkpoints inherit
  that restriction is a legal question we could not settle: treat commercial use of **all four** voices as
  **unverified** until you have your own answer.
* `irina`: the card says "Unknown". A third-party package listing (Arch Linux `rhvoice-voice-irina`) shows
  CC-BY-SA-4.0, but that is a secondary source, not the voice author. Do not rely on it.

## Install and run

The original repo is archived; the maintained fork ships a manylinux wheel, no build needed:

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install piper-tts          # 1.8.0 in our run; pulls onnxruntime + numpy (~70 MB)
python -m piper --help
```

Download the voices (`ru/ru_RU/<voice>/medium/ru_RU-<voice>-medium.onnx` and `.onnx.json`) from
`rhasspy/piper-voices` into `./voices/<voice>/`. The four `.onnx.json` configs are in `tts/piper/voices/`
(weights are not in this repository). Then:

```bash
pip install num2words
PIPER_VOICES_DIR=./voices OUT_DIR=./out python3 tts/piper/run_all.py
```

Telephone format (8 kHz mono mu-law): `ffmpeg -y -i in.wav -ar 8000 -ac 1 -f mulaw out-8k-ulaw.raw`.

## Hardware used for the measurements

CPU only. 8x Intel Xeon E5-2695 v2 @ 2.4 GHz (Ivy Bridge, no AVX2/FMA), 14 GB RAM, no GPU
(run of September 2026).

## Measured numbers

Source: `tts/piper/bench_results.json` (6 phrases x 4 voices, one process). **"speed" = audio seconds /
synthesis seconds (higher is faster)**, the inverse of RTF (RTF = 1 / speed).

| Voice | speed (audio/synth), range over 6 phrases | RTF equivalent |
|---|---|---|
| denis | 2.09-2.37x | 0.42-0.48 |
| dmitri | 1.54-2.45x | 0.41-0.65 |
| irina | 3.79-4.96x | 0.20-0.26 |
| ruslan | 1.64-3.60x | 0.28-0.61 |

* Memory: one loaded model peak RSS **239 MB** (isolated single-phrase run, `resource.getrusage`); all four
  models in one process **444 MB** (`peak_rss_mb` in the JSON).
* Model load time: an earlier isolated run measured 1.3-2.1 s for a cold load; the stored JSON run shows 3.3-6.3 s
  (`load_s`). The two runs disagree (the JSON run was probably made under load); treat load time as "a few
  seconds, once per process".
* First audio: `voice.synthesize(text)` yields one chunk per **sentence**, not sub-sentence. For a
  2-sentence phrase the first chunk arrived after 2.40 s (= full synthesis time of sentence 1).
* Telephony 8 kHz mu-law: moderate degradation, digits and addresses stayed intelligible (listening only).
* Phrases `privet` and `shanghai` were reworded for the public release; a re-run differs slightly.

## Quality notes for Russian

Findings from phoneme output (`--debug`), not from listening panels:

1. Plain integers are expanded correctly (`4521` -> "четыре тысячи пятьсот двадцать один").
2. **Space as thousands separator breaks:** `15 750 рублей` -> "пятнадцать семьсот пятьдесят" ("тысяч" lost).
3. **Times (`14:00`) and hyphenated ordinals (`25-го`) are not expanded.**
4. **Dates:** `3 сентября` gets a cardinal ("три"), not the genitive ordinal ("третьего").
5. **Homographs get a fixed stress:** "замок" is always "замОк", also in the "castle" sense.

`tts/piper/normalize_ru.py` (regex + `num2words`) fixes 2-4. Before/after on the sample sentence: `15 750` ->
"пятнадцать тысяч семьсот пятьдесят"; `3 сентября` -> "третьего сентября"; `25-го` -> "двадцать пятого".
Its limits: no case agreement after prepositions ("с четырнадцать ноль-ноль"), crude ordinal declension,
no noun-numeral agreement. For production use a full normalizer (e.g. `nemo_text_processing`).

Voice character (subjective, single listener): `irina` had an even pace with no audible clicks on the 6
phrases; the male voices sounded slightly duller on long phrases. All share one phonemizer.

## Known pitfalls

* Chunk = one sentence: for low latency split text into short sentences yourself.
* Keep the model in memory per worker; always normalize text (digits, times, dates) before synthesis.

## Verdict

Very light and fast (faster than real time on a 2013 CPU), but Russian text front end is
weak without normalization and the licensing of the Russian voices is the real constraint: `ruslan` is
non-commercial, `irina` is unlicensed on its card, and all four descend from a research-licensed base voice.
Fine for local/research use; do a legal review before shipping commercially.
