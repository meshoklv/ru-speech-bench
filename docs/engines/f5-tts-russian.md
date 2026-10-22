# F5-TTS, Russian fine-tune (Misha24-10/F5-TTS_RUSSIAN)

**Kind:** TTS, zero-shot voice cloning (needs the reference **transcript**), Russian + English, local GPU, flow-matching DiT.
**Upstream:** fine-tune <https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN> (further trained from `SWivid/F5-TTS` per its card,
"on 5 000 hours" of Russian and English); original <https://github.com/SWivid/F5-TTS>, <https://huggingface.co/SWivid/F5-TTS>.
(`Misha24-10` is the third-party author's HF handle, not related to this repository.) Tested: September 2026, `f5-tts` from PyPI.

## License (checked on the official pages)

| Part | License | Source |
|---|---|---|
| **Weights, the Russian fine-tune that was used** | **CC BY-NC 4.0 (`cc-by-nc-4.0`) - NON-COMMERCIAL** | HF card metadata of `Misha24-10/F5-TTS_RUSSIAN` (card has no separate licensing section) |
| Weights, original `SWivid/F5-TTS` | **CC BY-NC 4.0** | HF card metadata (card does not discuss the training-data license) |
| Code `SWivid/F5-TTS` | **MIT** ("Copyright (c) 2024 Yushen CHEN") | `LICENSE` in the GitHub repo |

**Commercial use of the weights: NO.** The fine-tune's license is the same as the base weights', checked on the fine-tune's own page.
The MIT code could be reused to train your own weights on your own data (not done here).

## Install & run

```bash
uv venv --python 3.12 venv-f5 && source venv-f5/bin/activate      # separate venv from XTTS
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124   # torch 2.6.0+cu124
pip install f5-tts soundfile
python tts/f5/run_f5.py --ckpt v2 --accent --refs ./refs/refs.json
```
Checkpoints used: `F5TTS_v1_Base/model_240000_inference.safetensors` (v1, stress marks optional) and
`F5TTS_v1_Base_v2/model_last_inference.safetensors` (trained on 100% stress-annotated text -> feed `+` marks).
**The vocab lives only in `F5TTS_v1_Base/vocab.txt` and is shared by every checkpoint**; without it the output is garbage.
Reference: 8-12 s of dense speech plus its exact transcript, in `./refs/refs.json` (your own audio, rights required).

## Hardware for the measurements

Windows 11, **NVIDIA RTX 3090 Ti 24 GB**, GPU inference, seed 1234. CPU and RAM were not recorded. Not run on CPU.

## Measured numbers

The script prints `rtf` as **audio/synth (higher = faster)**; derived usual RTF = 1/x.

| Metric | Value |
|---|---|
| Speed, v2 | audio/synth **3.00-5.50** (derived RTF ~0.18-0.33); 1.1-3.1 s per phrase |
| Time to first audio | **none (no streaming)**: the full phrase is produced first, 1.1-3.1 s of silence |
| VRAM | peak allocated 0.78-0.89 GB; `nvidia-smi` delta **+1.0 GB after load, +1.1 GB after synthesis** |
| Load | 10 s warm; 209 s first run including download |
| Round-trip STT (Gemini 2.5 Flash), WER excluding numerals | **v2 + stress: 3.2%** (0 of 10 files with real errors); v1_Base 11.7% (3 of 10); XTTS for reference 6.4% |
| Speaker similarity to the reference (XTTS speaker encoder, cosine) | v2: clone B **0.751**, clone A (phone-quality clip) **0.699**; v1_Base 0.749 / 0.680 |
| UTMOS (English-trained, relative use only) | v2 clone B 2.51, clone A 2.45; v1_Base 2.48 / 2.47 (references: B 2.56, A 1.46) |

## Quality notes for Russian

- **Stress marks are part of the format** for v2 / accent_tune: `Здр+авствуйте` gives correct stress, bare text gets guesses.
- Copies the reference voice more faithfully than XTTS, **including its recording character** (phone-codec sound), so it scores
  lower UTMOS but higher similarity; copies the reference **tempo** literally (a slow reference gave ~1.5x longer output than XTTS;
  `--speed` helps at the cost of naturalness).
- v1_Base leaked reference words into the output in 3 of 10 files (prepended "Всё", "Э-э, всё"); v2 with stress marks: 0 of 10 - **use v2**.
- Numbers from digits and 8 kHz telephone format were not tested. Accent and intonation were not listened to.
- Outputs hit 0.0 dBFS on some files; normalise before telephony.

## Known pitfalls

1. **A reference with pauses stretches the output.** F5 estimates the length as `ref_duration x (chars of text / chars of ref transcript)`;
   an 11 s clip with only ~50 characters of speech turned a phrase XTTS says in 6.2 s into 17.95 s. Use a window with ~100% speech
   (found with an ffmpeg `silencedetect` scan) and an exact transcript; then the same phrase took 9.6 s.
2. Wrong or default (English) vocab -> garbage.
3. Reference clip at 16 kHz is upsampled to 24 kHz without adding detail: caps clone quality.
4. Do not clone a clip without checking it: a file named like a studio recording turned out to be another TTS's output.
5. Streaming: the tested `f5_tts.api.F5TTS` exposed only `infer / transcribe / export_wav / export_spectrogram`. A later
   check of `infer_batch_process(streaming=True)` found only slicing of already-synthesized audio (see the ESpeech card).
6. Windows console codepage and SSH drops during model download, as for XTTS.

## Verdict

Most faithful voice copying and lowest word-error of the pair (with v2 + stress marks), half of XTTS's VRAM, but 1-3 s of silence
before each phrase and **non-commercial weights (CC BY-NC 4.0)**. Prototype / research only unless you train your own weights.
