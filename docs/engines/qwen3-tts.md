# Qwen3-TTS (12Hz-1.7B-Base)

**Kind:** TTS, zero-shot voice cloning, multilingual (10 languages incl. Russian), local GPU.
**Upstream:** code <https://github.com/QwenLM/Qwen3-TTS> (PyPI package `qwen-tts`),
weights <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base>.
Tested: September 2026, package `qwen-tts` with `transformers 4.57.3`.

## License (checked on the official pages)

| Part | License | Source |
|---|---|---|
| Code | **Apache License 2.0** | `LICENSE` in the GitHub repo ("Apache License, Version 2.0, January 2004") |
| Weights | **Apache-2.0** (declared) | HF model card metadata `license: apache-2.0` |

Commercial use: **allowed** (Apache-2.0). Caveat: the weights repo has no `LICENSE` file; the
license is declared only in the card metadata, the full text lives in the code repo.

## Install & run

```bash
python -m venv venv && source venv/bin/activate        # Python 3.12
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install qwen-tts soundfile numpy
```
Weights (4.54 GB: 3.86 GB model + 0.68 GB speech tokenizer) download on first run.
Put your own reference clip (>= 3 s per the card; clean speech you have rights to) and its
exact transcript into `./refs/` + `./refs/refs.json` (format: `tts/qwen3/common.py`).

```bash
python tts/qwen3/qwen_synth.py            # 5 phrases, RTF / VRAM
python tts/qwen3/qwen_bench.py --voice voiceA
python tts/qwen3/qwen_deep_bench.py --voice voiceA
python tts/qwen3/qwen_dialog.py           # two-speaker scene from tts/qwen3/jobs.json
```
Attention was `sdpa`; `flash-attn` is not installable as a wheel on Windows (the effect on speed was not measured).

## Hardware for the measurements

Windows 11, **NVIDIA RTX 3090 Ti 24 GB**, GPU inference, bfloat16. CPU model and system RAM were not recorded.
Not run on CPU.

## Measured numbers

RTF = synth time / audio time (lower is faster). Sources: raw run logs summarised in the original
test report; per-script output is the `RESULT` / `BENCH` lines of the scripts in `tts/qwen3/`.

| Metric | Value |
|---|---|
| RTF, 5 phrases, idle GPU, 3 repeats | **2.81-3.28** (about 3x slower than real time) |
| Deep benchmark (1 cold + 5 warm) | median RTF **2.868**, cold run only 0.9% slower; median 61 forward passes per phrase |
| Time per talker pass | **222 ms** (a 1.7B model on this card should need roughly 10-20 ms - estimate) |
| Average GPU utilisation during synthesis | **~31%** (max 39-84%) |
| Shortest phrase ("Здравствуйте!", 0.88 s audio) | 2.68-2.89 s to synthesize = floor on first-audio latency without streaming |
| Time to first model pass | 7 ms; so a streaming variant could in principle emit the first frame after ~230 ms (extrapolation, not measured) |
| VRAM | weights 4.20 GB; peak allocated 4.78-4.99 GB; torch reserved 6.11 GB; `nvidia-smi` **7.36-7.77 GB** |
| Model load (weights cached) | 6.25 s |
| Dialogue of 11 lines | ~2.5 min wall time (RTF 2.79-2.90 on the main speaker) |

## Quality notes for Russian

- **Intelligibility:** round-trip through STT (Gemini 2.5 Flash): 4 of 5 test phrases verbatim; the long
  number ("четыре тысячи пятьсот двадцать один") came back as "4 50 21" once (either STT or choppy reading;
  a later run on a different number was read correctly). Not listened to by a human in this test.
- **Stress:** no stress control and no accentuation helper (unlike ESpeech with RUAccent).
- **Cloning pitch fidelity:** median F0 of a 123.7 Hz male reference came out at **148.1 Hz** (+24 Hz);
  in the dialogue scene the cloned voice was 10-20 Hz too high. On a female reference: 166.7 Hz vs 172.7 Hz.
  This measures pitch only, not timbre.
- **Reference leakage (major):** the model tends to continue the reference: a stray last word of the
  reference text was prepended to the output (first batch), and in a 6-line scene 1-3 lines per run were
  damaged (swallowed greeting, prepended word). Ending the reference on a finished sentence reduced it
  (from 2-3 of 6 to 1 of 6) but did not remove it; very short lines were worst. Mitigation: re-roll seeds with an
  STT check (`tts/qwen_fix.py`, `tts/check_zv.py`), or `x_vector_only_mode=True` (not tested; cloning
  quality may drop).
- **Multilingual / emotion:** 10 languages; emotion control by instructions exists in the CustomVoice/VoiceDesign
  variants (not tested).
- Not in the public Russian TTS leaderboard that was consulted (it predates the model).

## Known pitfalls

1. The official package exposes **no streaming output** (only `generate_custom_voice`, `generate_voice_design`,
   `generate_voice_clone`, `create_voice_clone_prompt`; `non_streaming_mode=False` only simulates streamed *text*
   input). The model card advertises "97 ms" end-to-end latency and both streaming modes; that was **not
   reproducible with the tested package** (floor ~2.7 s). Newer package/serving stacks (vLLM-Omni) were not tried,
   and the card text may have changed since - treat as unverified.
2. Do not monkey-patch the talker's `forward` for profiling: `generate()` inspects its signature and fails. Use a
   forward pre-hook (see `qwen_deep_bench.py`).
3. Does not fit next to a 19 GB LLM on a 24 GB card (7.4-7.8 GB needed, ~5.7 GB were free) - arithmetic from
   separate measurements, not tested concurrently.
4. Re-synthesising with a different seed is the only practical fix for leakage, which is unusable in a live call.

## Verdict

Clear Russian speech and flexible multilingual cloning from ~3 s, permissive license, but **3x slower than
real time with the stock package**, no streaming, pitch drift in clones and frequent reference leakage:
not suitable for live telephony; usable for offline voice-over where each take is checked.
