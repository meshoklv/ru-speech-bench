# STT methodology: disagreement metric, telephone simulation, streaming latency

Used for the T-one and GigaAM measurements (`stt/bench_tone_*.py`, `stt/cloud-compare/`).

## 1. "Disagreement", not WER

There is no human-labelled reference, so the reference is another recognizer's output (GigaAM or a
cloud transcript). The score is word-level Levenshtein distance divided by the reference length,
after lower-casing, mapping "ё" to "е" and dropping case and punctuation. A score of 20% does **not**
mean one word in five is wrong: insertions, spelling variants and errors of the reference itself are
included. It is valid only for comparing variants of the same system (greedy vs beam, 16 kHz vs 8 kHz,
splitter settings). For comparing many engines, `stt/cloud-compare/analyze.py` uses the mean distance
to all other engines ("distance to consensus"), which rewards agreeing with the majority. A real WER
needs a hand-labelled reference.

## 2. Simulating a phone line

```bash
ffmpeg -i in.oga -ar 16000 -ac 1 -c:a pcm_s16le X-16k.wav
ffmpeg -i X-16k.wav -ar 8000 -ac 1 -c:a pcm_mulaw X-ulaw.wav     # G.711 mu-law
ffmpeg -i X-ulaw.wav -ar 8000 -ac 1 -c:a pcm_s16le X-phone8k.wav  # back to PCM
```
T-one downsamples every input to 8 kHz itself, so "16k" vs "phone8k" isolates the damage done by the
codec, not by the sample rate.

## 3. Streaming latency

Feed 300 ms chunks (`CHUNK_SIZE` samples at 8 kHz) through `pipeline.forward(chunk, state)`.
Two different quantities are measured:
- **Compute per chunk:** wall time of each `forward` call (mean, p50, p95, p99, max) and the share of
  chunks slower than 300 ms; if any is, the stream falls behind.
- **Emission lag:** audio position at the moment a phrase is returned minus the phrase's true end time
  (in audio seconds, not wall time). This is dominated by the splitter waiting for silence
  (`MIN_SILENCE_DURATION`, 20 frames of 30 ms = 0.6 s by default) plus the 300 ms chunk and a model
  bias, not by CPU speed.

## 4. Thread count and CPU cost

`intra_op_num_threads` in {1, 2, 4, 8}, best of 4 repetitions, wall time and `process_time` (CPU).
On shared hardware take the best repetition, otherwise neighbours' load dominates; log
`/proc/loadavg` next to every run. Single-thread was the cheapest per stream by a wide margin.

## 5. Concurrency

N threads, each with its own splitter and state but one shared model session and (for beam search)
one shared language model. Success criterion: the slowest line finishes before its audio ends, and
p95 chunk time stays far below 300 ms. Memory is paid once per process, not per line.

## 6. Routing doubtful phrases to a second recognizer

Per-phrase confidence = mean probability of the chosen symbol over non-blank frames. It separated a
clean clip (median 0.97, 0% of phrases under 0.80) from hard clips (0.89-0.91, 12-22% under 0.80), but
it was not shown that low confidence points at the wrong phrases (needs per-phrase labels), and some
errors are made confidently. The routing itself (T-one live, GigaAM re-reading the doubtful parts)
was proposed, not built.

## Caveats

Each result is from 4 clips (T-one) or 7 clips (cloud comparison) of private audio that is not
published; the splitter threshold was checked on 2 clips only; concurrency above 4 lines was not
measured. Re-run the scripts on your own audio before relying on a number.
