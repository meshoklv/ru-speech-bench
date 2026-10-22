# Smart Turn v3 (end-of-turn detection)

**Kind:** turn detection (audio-based "has the speaker finished?"), local CPU |
**Upstream:** https://huggingface.co/pipecat-ai/smart-turn-v3 , https://github.com/pipecat-ai/smart-turn

## What it is

A small (8M-parameter) audio model that looks at the last 8 s of speech and outputs the probability that
the speaker's turn is complete. It is a complement to a VAD: the VAD says "speech stopped", Smart Turn says
"and the sentence is finished". Tested file: `smart-turn-v3.2-cpu.onnx` (8.3 MB, int8).
Input `input_features [N, 80, 800]` (80 mel bands x 800 frames = 8 s at 16 kHz, from
`WhisperFeatureExtractor`), output `logits [N, 1]`. No torch needed, only `onnxruntime` and the feature
extractor.

## License of code and weights (checked 2026-10)

| Part | License | Commercial use | Source |
|---|---|---|---|
| Weights | **BSD 2-Clause** (HF metadata `bsd-2-clause`; "Copyright (c) 2024-2025, Daily") | yes | huggingface.co/pipecat-ai/smart-turn-v3 |
| Code | BSD 2-Clause (same project; the repository license file was not re-fetched) | yes (unverified for code) | github.com/pipecat-ai/smart-turn |

## Install and run

```bash
uv pip install onnxruntime transformers silero-vad torch numpy
# download smart-turn-v3.2-cpu.onnx from the HF repo into ./
SMART_TURN_MODEL=./smart-turn-v3.2-cpu.onnx python3 vad/smart-turn/st_lat.py     # latency
SMART_TURN_MODEL=./smart-turn-v3.2-cpu.onnx python3 vad/smart-turn/st_ab.py rec1.wav rec2.wav   # A/B test
SMART_TURN_MODEL=./smart-turn-v3.2-cpu.onnx python3 vad/smart-turn/st_test2.py rec1.wav ...      # labelled test
```

(`torch` and `silero-vad` are needed only by the test scripts that segment recordings.)

## Hardware used for the measurements

CPU only: Intel Xeon E5-2695 v2 @ 2.4 GHz (2013, no AVX2/FMA), 8 cores, 14 GB RAM. Latency measured on an
**idle** machine, warm-up 5 runs + 30 repeats.

## Measured numbers

Source: console output of the `vad/smart-turn/` scripts (September 2026); no JSON file was saved.

**Latency (our run):**

| intra_op threads | median | p95 |
|---|---|---|
| 1 | 367.8 ms | 385.5 ms |
| 2 | 198.3 ms | 218.4 ms |
| 4 | **109.4 ms** | 126.2 ms |
| 8 | 185.1 ms | 252.2 ms |

Mel feature extraction (`WhisperFeatureExtractor`): median 61.5 ms. Full cycle ~171 ms (61 + 109) vs the
~12 ms CPU figure the project advertises (that number is only in the blog-post URL slug, not in the model
card, and was presumably measured on a modern CPU). Peak RSS 425 MB, mostly torch used by the test scripts.

**Accuracy, upstream (reported, NOT reproduced):** the project's `benchmarks/smart-turn-v3.2-cpu.md` lists
Russian with 1,468 samples: accuracy 93.53 %, precision 0.920, recall 0.960, F1 0.940, FPR 4.36 %, FNR 2.11 %;
7th of 23 languages; overall model accuracy 92.63 % over 31,527 samples.

**Accuracy, our test (weak material):** no labelled dialogue set was available, so labels were derived from
7 real Russian voice messages (monologues). Two tests:

1. "finished" (pause > 3 s after speech) vs "pause then continues" (pause 0.25-0.8 s), both cut at VAD
   boundaries: 23/84 = 27.4 % overall, false "finished" 67.6-89.5 % per file. **This is a flaw of the test,
   not a verdict on the model:** median p(finished) was 0.87-0.91 in both groups, because in a monologue the
   speaker really does finish a sentence before breathing.
2. Controlled A/B on identical context: audio cut at a natural phrase end (A) vs 0.7 s earlier, mid-word (B),
   28 pairs: median p 0.904 (A) vs 0.645 (B); share of p > 0.5: 71 % vs 54 %; A > B in 79 % of pairs;
   **ROC-AUC 0.635**. The model separates the cases better than chance but says "finished" too readily at the
   0.5 threshold (even on mid-word cuts, 54 % of the time); the threshold should be raised on this material.

## Quality notes for Russian

Russian is explicitly in the upstream benchmark (93.53 %), but our monologue-based tests neither confirm
nor refute it. The material is not the model's training domain (human <-> voice-agent dialogue). A fair
check needs recorded real conversations labelled "finished / paused". See
`docs/methodology/turn-detection-eval.md`.

## Known pitfalls

* On CPUs without AVX2/FMA the advertised 12 ms is out of reach (~110 ms inference + ~60 ms features).
* Best thread count was 4 of 8 cores; 8 threads got slower.
* Default threshold 0.5 over-triggers on our material.
* Do not judge it on monologue voice messages.

## Verdict

Permissive license and a published Russian number, but on our data the signal is weak (AUC 0.635 on a
mid-word-cut test) and unproven; the decision needs a labelled set of real dialogues. Defer until then.
