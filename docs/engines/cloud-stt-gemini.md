# Cloud STT: Gemini chat models with audio input (via an API aggregator)

**Kind:** STT, cloud. **What was actually tested:** general-purpose Gemini models
(ids `gemini-3-5/3-6/3-7/3-8-flash-openai`, `gemini-3.1-pro`, `gemini-3-pro`) called through an API
aggregator (kie.ai) with the audio attached to a chat message. They are not dedicated
speech-to-text endpoints. Google's dedicated "Gemini 3.5 Transcribe" model was **not tested**: it was
not offered by the aggregator (calls returned "model not supported") and no Google API key with
billing was available.

## License / terms

Hosted service, no weights. Use is governed by Google's and the aggregator's terms of service,
which were not reviewed here: commercial use **unverified**. Audio leaves the machine.

## Install and run

No install; an API key in the environment (`AGGREGATOR_API_KEY`) and ffmpeg:
```bash
python stt/cloud-compare/run.py gemini-3.6-flash clip.ogg
```
Audio is converted to mono 16 kHz mp3 and sent as a data URI in the chat message
(`stt/cloud-compare/run.py`).

## Hardware

Remote; calls issued from an 8 vCPU / 24 GB VM, so timings include network and the aggregator queue.

## Measured numbers (7 private clips, 7-130 s; no reference transcript; see `../../stt/cloud-compare/SUMMARY.md`)

| Model | Succeeded | Distance to consensus | Median time |
|---|---|---|---|
| gemini-3.1-pro | 5/7 | 7.0% | 61 s |
| gemini-3.6-flash | 7/7 | 7.5% | 12 s |
| gemini-3.6-flash, strict prompt | 7/7 | 8.0% | 58 s |
| gemini-3.5-flash | 7/7 | 8.2% | 17 s |
| gemini-3.8-flash | 7/7 | 8.4% | 13 s |
| gemini-3-pro | 7/7 | 8.5% | 44 s |
| gemini-3.7-flash | 6/7 | 8.6% | 12.5 s |
| (local GigaAM-v3 for comparison) | 7/7 | 9.4% | 13 s |

Distance = mean word-level edit distance to the other engines, not WER. Timings come from overlapping
runs and are noisy. Cost: about 0.04 USD for about 55 calls (aggregator balance difference).

## Quality notes for Russian

- Latin-script terms and brand names are written better than by GigaAM, but spelling varies by model.
- LLM-style transcribers can soften profanity, drop or invent a word, or truncate the end; filler
  tokens differ widely (11 to 61 per engine over the 7 clips). Some keep hesitations (pro models).
- A stricter "do not guess" prompt did not help and made calls slower.

## Pitfalls

- Read timeouts at 180 s on the slowest models (3 of 56 calls); pro models are slow and flaky.
- Quiet fallback to a weaker engine when the cloud call fails hides the cause of "wrong" text.
- Model ids and availability on the aggregator change.

## Verdict

Slightly closer to the group's consensus and better with Latin terms than local GigaAM, at ~12 s per
file for the flash tier; no measured evidence that it is more accurate (no reference). Pro tier not
worth the latency here.
