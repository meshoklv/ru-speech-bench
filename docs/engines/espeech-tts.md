# ESpeech-TTS-1 (RL-V2, plus podcaster variant)

**Kind:** TTS, zero-shot voice cloning, Russian only, local GPU. F5-TTS architecture (DiT, flow matching),
used together with **RUAccent** (automatic stress marks).
**Upstream:** weights <https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2> (also `ESpeech/ESpeech-TTS-1_podcaster`),
inference code <https://github.com/SWivid/F5-TTS>, accentizer <https://github.com/Den4ikAI/ruaccent>.
Tested: September 2026.

## License (checked on the official pages)

| Part | License | Source |
|---|---|---|
| Weights `ESpeech-TTS-1_RL-V2` | **Apache-2.0** (declared) | HF card metadata `license: apache-2.0`; repo has 4 files, no separate `LICENSE` |
| Weights `ESpeech-TTS-1_podcaster` | **Apache-2.0** (declared) | HF card metadata |
| Training dataset `ESpeech/ESpeech-webinars2` | Apache-2.0 (declared; read at test time) | HF dataset card |
| Inference code F5-TTS | **MIT** ("Copyright (c) 2024 Yushen CHEN") | `LICENSE` in the GitHub repo |
| Vocoder `charactr/vocos-mel-24khz` | MIT (read at test time) | HF model card |

Commercial use: **allowed per the declared licenses.** Unverified: the model card does not state a base
checkpoint, and the original F5-TTS weights are CC-BY-NC-4.0 (see the F5-TTS card). If ESpeech was initialised
from those weights, upstream terms could matter; the authors' card does not say, so verify before commercial use.

## Install & run

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install f5-tts ruaccent soundfile       # no conflicts with qwen-tts in the same venv
```
Weights: `espeech_tts_rlv2.pt` (2 697 096 939 bytes) + `vocab.txt` from the HF repo into `./models/`
(or `tts/espeech/dl-espeech.ps1`, a resumable downloader for flaky links). Model config:
`dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4` (from the card). Vocoder downloads itself.
The **reference clip must be <= 12 s** (card) and needs an exact transcript. Put clips in `./refs/` + `./refs/refs.json`.

```bash
python tts/espeech/espeech_synth.py        # 5 phrases
python tts/zv_esp.py                       # dialogue scene
python tts/esp_stream_bench.py --voice voiceA
```

## Hardware for the measurements

Windows 11, **NVIDIA RTX 3090 Ti 24 GB**, GPU inference (`nfe_step=32`, `cross_fade_duration=0.15`, `speed=1.0`).
CPU and RAM were not recorded. Not run on CPU.

## Measured numbers

RTF = synth / audio (lower is faster).

| Metric | Value |
|---|---|
| RTF, 5 phrases (female ref) | **0.174-0.362**; cloned reference voice 0.288; dialogue lines 0.21-0.52 |
| Synth time per phrase | 1.55-2.73 s for 5.2-11.3 s of audio |
| Load | model 2.82 s, vocoder 0.60 s, RUAccent 6.33 s |
| VRAM | 0.74 GB after load; peak allocated 0.82-0.84 GB; `nvidia-smi` **2 808 MB** |
| Weights on disk | 2.70 GB |
| Round-trip STT (Gemini 2.5 Flash) | **6 of 6** test lines verbatim; the number came back as "4 521" |
| Pitch fidelity of cloning | 123.7 Hz reference -> **120.6 Hz** (3 Hz off); dialogue 113.7-123.7 Hz vs 114.3 Hz reference |
| Streaming (`infer_batch_process(streaming=True, chunk_size=2048)`) | one segment: first chunk **1 482 ms** = total time (RTF 0.234); text split into 3 phrases: first chunk **1 117 ms**, total 3.37 s (RTF 0.54) |
| 11-line dialogue | ~19 s wall time |
| Public leaderboard (ESpeech authors, `open_tts_leaderboard_ru`, `data.json`) | RL-V2: utmos 3.265, cer 1.4, sim 0.833; podcaster utmos 3.294 (figures as read at test time, not re-checked) |

## Quality notes for Russian

- **Stress:** RUAccent (`turbo3.1`) puts stress automatically (`+` before the stressed vowel), including numerals
  ("пятьс+от", "семьс+от пятьдес+ят") and "сентябр+я"; you can set `+` by hand and the automation skips that text.
- **Numbers:** read correctly when written out in words.
- **Cloning:** pitch matches the reference closely; reference leakage still occurs: in a 6-line scene 3 of 6
  lines needed a re-roll on the first pass (garbled line, repeated reference words, stray word). Fixed by another seed
  and by trimming at a pause; both fixes are offline-only.
- **Liveliness is not measured by any number here:** utmos and STT ignore intonation. A conversational (spontaneous)
  reference and filler words in the text ("э-э", "ну") were tried (`live_synth.py`, `zvonok_synth.py`), as well as a grid
  over `cfg_strength` / `nfe_step` / sway sampling (`grid_synth.py`) and the podcaster checkpoint (`cv_pod_synth.py`);
  judgements were by listening and are not reproduced as numbers.
- In the authors' leaderboard a clean-sounding classic Russian TTS (Piper "Irina", utmos 3.672) outranks ESpeech RL-V2
  (3.265): utmos rewards cleanliness, not liveliness.

## Known pitfalls

1. `torchaudio` >= 2.9 reads/writes audio through `torchcodec`, which needs FFmpeg 4-7 shared libraries; where they are
   missing `import torchcodec` works but the first real read fails ("Could not load libtorchcodec"). Fix used:
   `tts/espeech/common.py::patch_audio_io` swaps in `soundfile`.
2. Reference > 12 s is out of spec; keep clips short and ending on a complete sentence.
3. "Streaming" slices already-synthesized audio (the whole text segment is generated first); only splitting the text
   lowers time-to-first-chunk, at the cost of total time.
4. `HF_HOME` should point at the disk you intend to use; Hugging Face downloads on the test machine ran ~0.9 MB/s and
   often stalled (hence the resumable downloader).
5. Weights re-appear in the cache if you keep running the engine after deleting them.

## Verdict

Best speed/VRAM/accuracy trade-off among the open Russian cloners tested: 3-6x faster than real time on 2.8 GB, correct
stress and numbers, close pitch cloning, permissive declared license. Remaining problems: ~1.1-1.5 s before the first
audio and occasional reference leakage; liveliness needs listening tests.
