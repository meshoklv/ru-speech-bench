# Evaluating end-of-turn detection without a labelled dialogue set

Context: Smart Turn v3 (`docs/engines/smart-turn-v3.md`) was tested on monologue voice messages, because no
labelled telephone dialogues were available.

## Approach 1: segment with VAD, label by the following pause (weak)

`vad/smart-turn/st_test2.py`. Cut recordings into speech segments with Silero VAD; keep the audio up to the
segment end plus a short silent tail (`TAIL_MS`, default 200 ms), at most the last 8 s.

* "finished" = pause after the segment > 3 s;
* "continues" = pause 0.25-0.8 s.

Result in our run: 27.4 % accuracy and 68-90 % false "finished". Reading: **this labelling is a poor proxy for
monologues**, where speakers finish a sentence before every breath, so both groups look "finished"
(median p 0.87-0.91 in both). It cannot be used to estimate model accuracy.

## Approach 2: controlled A/B on identical context (better)

`vad/smart-turn/st_ab.py`. For every speech segment followed by a >= 1 s gap and at least 1.5 s long, build
two inputs from the same audio: A ends 200 ms after the natural segment end, B ends 700 ms *before* it
(mid-word). A model that detects completion should score A above B. Report median p, share with p > 0.5,
share of pairs with A > B, and ROC-AUC over the pooled A/B scores (0.5 = chance).

Our result: A > B in 79 % of 28 pairs, AUC 0.635, and p > 0.5 in 54 % of the mid-word cuts, i.e. a threshold
above 0.5 would be needed on this material.

## What a proper evaluation needs

Recorded real conversations with human labels at each pause ("turn is over" / "will continue"), from the
target domain and channel (telephone, 8 kHz). Until then, treat upstream-published numbers as claimed and
local numbers as "not confirmed on monologues".

## Latency measurement tips

Measure on an idle machine, with warm-up, 30+ repeats, report median and p95, sweep `intra_op_num_threads`
(`vad/smart-turn/st_lat.py`; more threads is not always faster: 4 beat 8 on an 8-core box), and report feature
extraction separately from the network.
