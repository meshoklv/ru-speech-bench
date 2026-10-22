# Third-party notices

This repository contains **no third-party model weights, audio or datasets**. It contains scripts that download or call the engines below, and
notes about them. Each engine keeps its own license. The table is a summary of what was read on the upstream pages in September-October 2026
(date and source in each card). It is not legal advice; check the upstream page again before you ship.

Items marked **NON-COMMERCIAL** must not be used in a commercial product. `unverified` means we could not settle it from the upstream text.

| Engine | License (as read upstream) | Commercial use | Card |
|---|---|---|---|
| Fish Audio (API, s2.1-pro-free) | Hosted API: vendor ToS (not re-verified); open S1 weights CC-BY-NC-SA-4.0; fish-speech code: Fish Audio Community License v1.0 | unverified | [fish-audio.md](docs/engines/fish-audio.md) |
| kie.ai TTS (Gemini 2.5 Pro / 3.1 Flash TTS) | Hosted aggregator; terms of kie.ai and Google | unverified | [kie-ai-tts.md](docs/engines/kie-ai-tts.md) |
| CosyVoice 3 (Fun-CosyVoice3-0.5B) | Apache-2.0 (weights, per HF model card) | yes | [cosyvoice3.md](docs/engines/cosyvoice3.md) |
| IndexTTS-2 | bilibili Model Use License Agreement (custom) | unverified | [indextts2.md](docs/engines/indextts2.md) |
| GigaAM-v3 (sherpa-onnx RNN-T, punct) | MIT (code and weights) | yes | [gigaam-v3.md](docs/engines/gigaam-v3.md) |
| T-one (T-Tech) | Apache-2.0 (code and weights) | yes | [t-one.md](docs/engines/t-one.md) |
| sherpa-onnx (runtime) | Apache-2.0 | yes | [sherpa-onnx.md](docs/engines/sherpa-onnx.md) |
| Gemini chat models with audio (via aggregator) | hosted service; terms not reviewed | unverified | [cloud-stt-gemini.md](docs/engines/cloud-stt-gemini.md) |
| Piper (ru_RU-denis-medium) | Runtime piper1-gpl GPL-3.0; voice dataset CC0; fine-tuned from lessac (research-only data) | unverified | [piper.md](docs/engines/piper.md) |
| Piper (ru_RU-dmitri-medium) | Runtime piper1-gpl GPL-3.0; voice dataset CC0; fine-tuned from lessac (research-only data) | unverified | [piper.md](docs/engines/piper.md) |
| Piper (ru_RU-irina-medium) | Runtime piper1-gpl GPL-3.0; voice dataset RHVoice, license 'Unknown' on the model card | unverified | [piper.md](docs/engines/piper.md) |
| Piper (ru_RU-ruslan-medium) | Runtime piper1-gpl GPL-3.0; voice dataset Ruslan corpus CC BY-NC-SA 4.0 | **NO (non-commercial)** | [piper.md](docs/engines/piper.md) |
| Kokoro-82M (stock, Russian via espeak-ng) | Apache-2.0 (weights, code); kokoro-onnx MIT | yes | [kokoro.md](docs/engines/kokoro.md) |
| Kokoro-RU fine-tune (zaakirio/kokoro-ru) | Weights OpenRAIL; code Apache-2.0 | unverified | [kokoro.md](docs/engines/kokoro.md) |
| Chatterbox Multilingual (V2 weights) | MIT (code and weights); Perth watermark in every output | yes | [chatterbox.md](docs/engines/chatterbox.md) |
| Silero VAD (6.2.1) | MIT | yes | [silero-vad.md](docs/engines/silero-vad.md) |
| Smart Turn v3.2 (cpu onnx) | BSD 2-Clause | yes | [smart-turn-v3.md](docs/engines/smart-turn-v3.md) |
| Qwen3-TTS 12Hz-1.7B-Base | Apache-2.0 (code and weights) | yes | [qwen3-tts.md](docs/engines/qwen3-tts.md) |
| ESpeech-TTS-1 RL-V2 (F5-TTS arch + RUAccent) | Apache-2.0 (weights, declared); code MIT | yes | [espeech-tts.md](docs/engines/espeech-tts.md) |
| XTTS-v2 | Coqui Public Model License (weights, non-commercial incl. outputs); code MPL-2.0 | **NO (non-commercial)** | [xtts-v2.md](docs/engines/xtts-v2.md) |
| F5-TTS Russian fine-tune (Misha24-10, v2 + stress marks) | CC BY-NC 4.0 (weights, fine-tune and base); code MIT | **NO (non-commercial)** | [f5-tts-russian.md](docs/engines/f5-tts-russian.md) |

## Python packages used by the scripts (installed by you, not vendored)

| Package | License |
|---|---|
| piper-tts (OHF-Voice/piper1-gpl) | GPL-3.0-or-later |
| num2words | LGPL |
| kokoro-onnx | MIT |
| onnxruntime | MIT |
| sherpa-onnx | Apache-2.0 |
| T-one (`tone`) | Apache-2.0 |
| pyctcdecode, huggingface-hub | Apache-2.0 |

The reference voices used in the original runs are not distributed. Where to find legally usable reference voices:
[docs/methodology/liveliness-and-brightness.md](docs/methodology/liveliness-and-brightness.md).
