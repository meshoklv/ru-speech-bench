# T-one (T-Tech / T-Bank) - streaming Russian telephony ASR

**Kind:** STT, local, CPU-only (ONNX Runtime). **Upstream:** https://github.com/voicekit-team/T-one
(code), https://huggingface.co/t-tech/T-one (weights, org `t-tech`).

A streaming CTC Conformer (about 71.7 M parameters per the model card), trained on over 80,000 hours
of Russian speech, mostly telephony (8 kHz). It consumes audio in 300 ms chunks and finds phrase
boundaries itself.

## License (checked against the official pages)

| Part | License | Source |
|---|---|---|
| Code (GitHub repo) | Apache License 2.0 | repo README: "The code and models in this repository are released under the Apache 2.0 License" |
| Weights (`t-tech/T-one`) | Apache-2.0 | HF model card metadata `license: apache-2.0` |
| `kenlm.bin` language model (5.4 GB, optional) | unverified separately; shipped in the same HF repo | - |

Commercial use: **allowed** (Apache-2.0). The kenlm library used by the beam-search decoder has its
own license, which was not checked here.

## Hard limitation: no Latin letters, no digits, no punctuation

The output alphabet is the 33 Russian letters plus space (plus a CTC blank): in `tone/decoder.py`
`LABELS = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя '`. The model can **not** emit "Excel", "Kubernetes",
"4417" or any punctuation; it writes what it hears with Cyrillic letters ("12 500" becomes words;
a Latin brand name becomes a phonetic Cyrillic guess). A language model does not fix this - the
symbols are missing from the vocabulary. Only fine-tuning, or a second recognizer on doubtful
segments, can help.

## Install and run (as done for the measurements)

```bash
git clone --depth 1 https://github.com/voicekit-team/T-one.git
uv venv --python 3.12 .venv-tone && source .venv-tone/bin/activate
# greedy-only install (no compiler needed): install deps by hand, the package without deps
uv pip install "numpy>=1.17.3,<3.0.0" "huggingface-hub>=0.14.0,<1.0.0" \
               "onnxruntime>=1.12.0,<2.0.0" "pyctcdecode>=0.2.0,<1.0.0" "miniaudio>=1.50,<2.0"
uv pip install --no-deps -e ./T-one
```
```python
from tone import StreamingCTCPipeline
from tone.decoder import DecoderType
pipe = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)  # default is beam search
```
Benchmarks: `../../stt/bench_tone_*.py` (clips from `$TESTAUDIO_DIR`); method in
`../methodology/stt-disagreement-and-streaming.md`.

Recommended configuration found by measurement:
`ort.SessionOptions().intra_op_num_threads = 1` (+ `inter_op_num_threads = 1`) and
`StreamingLogprobSplitter.MIN_SILENCE_DURATION = 6`.

## Hardware used

CPU only: an 8 vCPU / 24 GB virtual machine (container shared with another busy container; host
`loadavg` was 8.9-19.3 during some runs, which is why the multi-thread table below is noisy). No GPU.
Python 3.12, onnxruntime.

## Measured numbers (4 private clips of 53 / 79 / 193 / 285 s; results in `../../stt/results_*.json`)

Speed (greedy, 1 ORT thread, clip a53, 53 s, best of 4): RTF 0.309 (x3.2 real time), 16.4 s CPU.

| ORT threads | wall | RTF | CPU time spent |
|---|---|---|---|
| 1 | 16.39 s | 0.309 | 16.4 s |
| 2 | 11.39 s | 0.215 | 22.2 s |
| 4 | 12.78 s | 0.241 | 41.0 s |
| 8 (library default) | 21.80 s | 0.411 | 105.1 s |

(`results_threads.json`.) The library's default session uses all cores and is **slower and about 6x
more CPU-hungry** than 1 thread.

Streaming, 300 ms chunks, 1 thread, 8 kHz telephone-style clips (`results_stream_1thread.json`,
`results_stream_beam.json`): mean 99-123 ms per chunk, p95 104-175 ms; 0 of 1091 chunks over 300 ms
with greedy; beam: 0.2% over 300 ms on the longest clip (max 301.7 ms). Phrase emission lag
1.05-1.09 s mean with the default splitter (1.17-1.20 s max); with `MIN_SILENCE_DURATION=6` it drops
to 0.64-0.65 s mean (0.78 s max) with no loss on the two clips tested (`results_silence.json`).
At 2 it breaks (words are cut: disagreement 20% to 46%, 41.5% to 64.6%).

Concurrency with the shared LM (beam, one process): 1/2/4 lines all keep real time, worst p95
about 110 ms, 0% chunks over 300 ms, RSS 5.18-5.19 GB (`results_parallel.json`). More than 4 lines
was not measured.

Memory: about 600 MB (greedy), about 5.2 GB with the kenlm LM; load 7.0 s. Download of the full set
(model.onnx 144 MB + kenlm.bin 5.46 GB + safetensors 287 MB) took 29 minutes.

## Quality (disagreement with another recognizer, NOT WER)

No human reference exists; the reference was GigaAM (clip a53) / a cloud transcript (clip b79). Numbers
are word-level edit distance, only usable to compare T-one variants with each other
(`results_disagreement*.json`).

| clip | greedy 16k | greedy 8k phone | beam+LM 16k | beam+LM 8k phone |
|---|---|---|---|---|
| a53 (clean) | 24.5% | 22.0% | 22.5% | 20.0% |
| b79 (two speakers, quiet) | 46.8% | 43.8% | 45.4% | 41.5% |

- Telephone codec (G.711 mu-law round trip) did not hurt: 8 kHz was equal or slightly better.
- The LM turns letter noise into real words but sometimes into the wrong plausible words
  ("filters in Excel" became something like "to drain which as in a village"); gain only 1.4-2.3
  points. A smooth wrong sentence is harder to spot than garbage.
- Vendor-published comparison (GigaAM v3 model card, not measured here): average WER 19.4 for
  "T-One + LM" vs 8.4 for GigaAM-v3 RNN-T across their domains.
- Confidence signal (mean probability of non-blank frames): median 0.97 on the clean clip, 0.89-0.91
  on hard ones; share of phrases below 0.80: 0% vs 12-22% (`results_confidence.json`). Shows the
  clip is hard, not which phrase is wrong (not verified per phrase).

## Pitfalls

- Default decoder is beam search: `from_hugging_face()` fails without kenlm; pass `DecoderType.GREEDY`.
- `kenlm` has no wheels on PyPI (sdist only; 0.3.0 needs cmake and a compiler); `pypi-kenlm` does
  not build on Python 3.12. It was built without root using the `ziglang` wheel as the compiler,
  uv's standalone Python (for headers), the cmake step removed from `setup.py`, and compiler wrappers
  dropping the unsupported `-Wl,--exclude-libs` flag. The upstream README says kenlm is not
  supported on Windows.
- Any input is downsampled to 8 kHz inside `read_audio`.
- The upstream web demo binds `0.0.0.0:8080` without authentication - bind to 127.0.0.1 if you use it.

## Verdict

The only engine here that streams on plain CPU with ~0.65 s phrase lag and no GPU, and 8 kHz
telephony is its home turf; text quality is clearly below GigaAM-v3 and Latin/digits are impossible.
Use it as the live "ears" of a phone bot and re-read the recording with a better recognizer.
