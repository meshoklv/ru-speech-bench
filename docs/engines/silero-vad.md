# Silero VAD

**Kind:** VAD (voice activity detection), local CPU | **Upstream:** https://github.com/snakers4/silero-vad |
PyPI: `silero-vad`

## What it is

A tiny neural voice-activity detector (JIT and ONNX variants shipped inside the pip package: `silero_vad.jit`
2.2 MB, `silero_vad.onnx` 2.2 MB, `silero_vad_16k_op15.onnx` 1.2 MB). It supports 8 kHz and 16 kHz audio and
works on ~32 ms chunks (512 samples at 16 kHz, 256 at 8 kHz). Upstream says it was trained on corpora of
"over 6000 languages", so it is not Russian-specific. Used here to cut recordings into speech segments and as
the segmentation step for the turn-detection tests.

## License of code and weights (checked 2026-10)

| Part | License | Commercial use | Source |
|---|---|---|---|
| Code and weights (in the pip package) | **MIT** (README: "permissive license (MIT)"; package classifier `License :: OSI Approved :: MIT License`, `LICENSE` inside the package) | yes | github.com/snakers4/silero-vad |

One README badge's alt text reads "License: CC BY-NC 4.0", which contradicts the MIT text, the sidebar and the
package metadata. We treat MIT as authoritative but this is a documentation inconsistency worth knowing about.

## Install and run

```bash
uv pip install silero-vad          # 6.2.1 in our run; weights ship inside the package
python3 vad/silero_test.py recording.wav 16000      # segmentation + per-chunk latency
python3 vad/bench_idle.py                           # torch vs ONNX, 16 kHz vs 8 kHz
```

For production use `load_silero_vad(onnx=True)`.

## Hardware used for the measurements

CPU only: Intel Xeon E5-2695 v2 @ 2.4 GHz (no AVX2/FMA), 8 cores, 14 GB RAM. Latency numbers: idle machine,
1 thread. The whole-file runs used 4 threads and were made under heavy machine load (load average ~21), so
the "whole file" times are an upper bound.

## Measured numbers

Source: console output of `vad/silero_test.py` / `vad/bench_idle.py` (September 2026); no JSON file was saved. Input: a 284.6 s real Russian voice message
(not distributed).

| Sample rate | Segments | Speech | Whole-file processing |
|---|---|---|---|
| 16 kHz | 101 | 158.6 s (56 %) | 9.8 s (x29 faster than real time) |
| 8 kHz | 98 | 142.6 s (50 %) | 8.1 s (x35) |

First segments at 16 kHz and 8 kHz were almost identical (e.g. (0.0, 2.0), (2.1, 4.0), (4.8, 5.5) vs
(0.0, 1.9), (2.1, 4.0), (4.8, 5.5)), so telephone-band audio does not degrade segmentation.

Per-chunk (32 ms) latency, idle machine, 1 thread:

| Backend | Rate | median | p95 | x real time |
|---|---|---|---|---|
| torch | 16 kHz | 1.124 ms | 1.217 ms | 28 |
| torch | 8 kHz | 0.952 ms | 1.004 ms | 34 |
| ONNX | 16 kHz | 0.577 ms | 0.637 ms | 55 |
| ONNX | 8 kHz | 0.513 ms | 0.562 ms | 62 |

Memory: 417 MB peak RSS, but that includes the loaded torch; the model itself is a few MB (pure
onnxruntime would use far less; not measured).

## Quality notes for Russian

Only the segmentation of one long Russian voice message was examined (counts and boundaries above); no
labelled frame-level accuracy (precision/recall) was computed, so no quality metric is claimed. It was also
used to pick a clean 12 s speech excerpt for voice-cloning references, which worked as intended.

## Known pitfalls

* In silero-vad 6.x the argument order is `get_speech_timestamps(audio, model)`, not `(model, audio)`. The
  wrong order fails with an unhelpful `TypeError: Audio cannot be casted to tensor. Cast it manually`.
* The 417 MB RSS figure was measured with torch loaded; it is not the model's own footprint.

## Verdict

Take it: MIT, ~0.5 ms per 32 ms chunk on a 2013 CPU with ONNX (one thread can serve dozens of streams), and
8 kHz audio segments as well as 16 kHz. Frame-level accuracy on Russian was not measured here.
