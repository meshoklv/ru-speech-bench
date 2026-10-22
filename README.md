<p align="center">
  <img src="docs/cover.png" alt="ru-speech-bench" width="100%">
</p>

<h1 align="center">ru-speech-bench</h1>

<p align="center"><b>Hands-on benchmark of 21 speech engines for Russian: TTS, STT and turn detection, with the scripts, the raw numbers and the licenses.</b></p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square">
  <img alt="Language" src="https://img.shields.io/badge/lang-Russian-red?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square">
  <img alt="Engines" src="https://img.shields.io/badge/engines-21-8A2BE2?style=flat-square">
  <img alt="PRs welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square">
</p>

<p align="center">
  <a href="#results">Results</a> ·
  <a href="#licenses-read-this-before-you-ship">Licenses</a> ·
  <a href="#how-to-reproduce">Reproduce</a> ·
  <a href="#methodology">Methodology</a> ·
  <a href="README.ru.md">Русский</a>
</p>

---

## Why

Benchmarks of speech models are mostly English, and the Russian ones rarely say on which hardware a number was taken, which
license the weights carry, or what broke while installing. This repo is the log of a real project that needed a
Russian phone-grade voice agent ([ru-voice-agent](https://github.com/meshoklv/ru-voice-agent)): every engine was installed,
run on Russian text, measured, and written up with its pitfalls.

- **Numbers only from real runs**, each with the exact hardware. "Not measured" is written where it was not measured.
- **Licenses checked against the upstream page** (date in every card). Non-commercial weights are flagged, not hidden.
- **Scripts included**: every number can be reproduced on your machine with your own reference voices and recordings.

> [!NOTE]
> The measurements were taken in September 2026 on two machines: an 8-vCPU Xeon E5-2695 v2 VM (CPU only, no AVX2) and a desktop
> with an RTX 3090 Ti (Windows 11). Numbers are comparable **within** a table row's hardware, not across rows.

## Results

### Text-to-speech

"Speed" is reported the way each run measured it; RTF = synthesis time / audio time (below 1 is faster than real time). Where the
original run logged audio/synthesis, the card says so and the value here is converted.

| Engine | Runs on | Speed (measured) | Russian quality notes | Weights license, commercial use |
|---|---|---|---|---|
| [Piper](docs/engines/piper.md) (denis / dmitri / irina / ruslan) | CPU, 8 vCPU Xeon | RTF 0.20-0.65 | even pace, no clicks (subjective); needs number/date normalisation | voice data: CC0 (denis, dmitri), unknown (irina), **CC BY-NC-SA (ruslan)**; runtime GPL-3.0; base-model caveat: unverified |
| [Kokoro-82M](docs/engines/kokoro.md) (Russian via espeak) | CPU 8 threads / 5950X / 3090 Ti | RTF 0.76 (Xeon), 0.2 (5950X), 0.05 (GPU) | intelligible, lively, no official Russian, stress sometimes wrong | Apache-2.0, yes |
| [Kokoro-RU fine-tune](docs/engines/kokoro.md) | 5950X CPU | ~1.2 s per 7 s of audio | authors report 2.50 % WER (not reproduced) | OpenRAIL, unverified |
| [Chatterbox Multilingual](docs/engines/chatterbox.md) | CPU 8 vCPU Xeon | RTF 22-24 (unusable on CPU) | words fine, digits garbled; clone cosine 0.80-0.82 | MIT, yes (watermarked output) |
| [Qwen3-TTS 1.7B](docs/engines/qwen3-tts.md) | RTX 3090 Ti | RTF 2.8-3.3 | 4/5 phrases verbatim, clone leaks the reference | Apache-2.0, yes |
| [ESpeech-TTS-1 RL-V2](docs/engines/espeech-tts.md) | RTX 3090 Ti | RTF 0.17-0.36, first chunk 1.1-1.5 s | 6/6 lines verbatim, automatic stress, close clone | Apache-2.0 declared, base-model license unverified |
| [XTTS-v2](docs/engines/xtts-v2.md) | RTX 3090 Ti | RTF ~0.5, first audio 0.47-0.53 s (streaming) | WER 6.4 % (no numerals), clone sim 0.58-0.69 | **CPML non-commercial** |
| [F5-TTS Russian fine-tune](docs/engines/f5-tts-russian.md) | RTX 3090 Ti | RTF ~0.2-0.3, 1.1-3.1 s to first audio | WER 3.2 % (no numerals), best clone 0.70-0.75 | **CC BY-NC 4.0** |
| [Fish Audio](docs/engines/fish-audio.md) (cloud API) | cloud | first audio ~1.0 s best, 1.7-3.7 s typical, up to 32.9 s | livelier than local voices (pitch spread 4.71 vs 2.14 st) | API terms: unverified; open weights **CC-BY-NC-SA** |
| [kie.ai TTS](docs/engines/kie-ai-tts.md) (Gemini TTS via aggregator) | cloud | 6-22 s per phrase, no streaming | intelligible | aggregator + Google terms: unverified |
| [CosyVoice 3](docs/engines/cosyvoice3.md) | RTX 3090 Ti | not measured | Russian not proven | Apache-2.0, yes |
| [IndexTTS-2](docs/engines/indextts2.md) | none | not run | no Russian text front end | **bilibili custom license** |

### Speech-to-text

There is no human-labelled reference in this repo, so the quality column is **disagreement between engines, not WER**.

| Engine | Runs on | Speed | Quality | License |
|---|---|---|---|---|
| [T-one](docs/engines/t-one.md) | CPU, 1 thread | RTF 0.31 offline, streaming 300 ms chunks in ~100-123 ms | disagreement 20-22 % clean (vendor WER 19.4 % with LM); no Latin letters or digits in output | Apache-2.0, yes |
| [GigaAM-v3](docs/engines/gigaam-v3.md) (via [sherpa-onnx](docs/engines/sherpa-onnx.md)) | CPU, 4 threads | median 12.8 s per 7-130 s file under load | distance to consensus 9.4 % (largest of 8, not WER); vendor WER 8.4 % | MIT, yes |
| [Gemini models with audio](docs/engines/cloud-stt-gemini.md) via aggregator | cloud | flash 12-17 s, pro 44-61 s per file | 7.0-8.6 % distance to consensus over 7 clips | hosted: unverified |

Full cloud-vs-local table: [stt/cloud-compare/SUMMARY.md](stt/cloud-compare/SUMMARY.md).

### Voice activity / end-of-turn

| Engine | Runs on | Speed | Notes | License |
|---|---|---|---|---|
| [Silero VAD 6.2.1](docs/engines/silero-vad.md) | CPU, 1 thread | 0.51-0.58 ms per 32 ms chunk (55-62x real time) | 8 kHz and 16 kHz give near-identical boundaries | MIT, yes |
| [Smart Turn v3.2](docs/engines/smart-turn-v3.md) | CPU, 4 threads | ~171 ms per decision | needs labelled dialogue data to tune; default threshold over-triggered on our material | BSD-2-Clause, yes |

## Licenses: read this before you ship

| Engine | Verdict |
|---|---|
| XTTS-v2, F5-TTS (Russian fine-tune), Fish open weights, Piper `ruslan` voice | **non-commercial** |
| IndexTTS-2 | custom restrictive license |
| Piper (all voices) | runtime is GPL-3.0; all four Russian voices are fine-tuned from a research-only English voice: **commercial use unverified, get your own legal answer** |
| Fish API, kie.ai, hosted Gemini | governed by the vendor's current terms, not re-verified |
| T-one, GigaAM-v3, Kokoro, Qwen3-TTS, Chatterbox, Silero VAD, Smart Turn, CosyVoice 3 | permissive (MIT / Apache-2.0 / BSD-2) as stated on the upstream pages; see each card |

Every card names the page the license was read from and the date. Not legal advice. Details: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## How to reproduce

No audio, weights or reference voices ship with the repo. Put your own (that you have the rights to) in `./refs/` and `./models/`, then
run the script listed in each engine card. For example, Piper on CPU:

```bash
pip install piper-tts num2words
# download a voice from https://huggingface.co/rhasspy/piper-voices (ru/ru_RU/<voice>/medium)
PIPER_VOICES_DIR=./voices OUT_DIR=./out python3 tts/piper/run_all.py
```

T-one streaming benchmarks and GigaAM server: [docs/engines/t-one.md](docs/engines/t-one.md), [docs/engines/gigaam-v3.md](docs/engines/gigaam-v3.md).

> [!WARNING]
> Scripts were cleaned for publication and compile (`python -m compileall`), but most were **not re-run** after the cleanup.
> Numbers in the tables come from the original runs, before the cleanup. Expect to adjust paths and model names.

## Methodology

- [Liveliness and brightness of a voice](docs/methodology/liveliness-and-brightness.md): pitch spread in semitones and the
  high-frequency band, to compare local and cloud voices without arguing about taste
- [TTS round-trip checks](docs/methodology/tts-roundtrip-checks.md): synthesise, transcribe, compare words; speaker-embedding cosine
- [Voice-cloning measurement](docs/methodology/tts-cloning-measurement.md): pitch, similarity, UTMOS, reference leakage
- [Turn-detection evaluation](docs/methodology/turn-detection-eval.md)
- [STT disagreement and streaming](docs/methodology/stt-disagreement-and-streaming.md): why "disagreement" is not WER

## Layout

```
docs/engines/    one card per engine: license, install, hardware, numbers, pitfalls
docs/methodology/ how things were measured
tts/ stt/ vad/ bench/   the scripts, grouped by engine
stt/cloud-compare/      cloud-vs-local STT run (scripts, timings, summary)
```

## Contributing

Add an engine: copy a card from `docs/engines/`, fill the license with a link and a date, report hardware and the command. PRs with
numbers from other hardware are very welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Code and docs: [MIT](LICENSE). The engines keep their own licenses, see above.

Made by [@meshoklv](https://github.com/meshoklv)
