# Cloud STT (Gemini models via an API aggregator) vs local GigaAM-v3 - small consistency check

**This is not a WER benchmark.** There is no human-made reference transcript. The "distance to
consensus" below only shows how far an engine's text is from the other engines' texts; it can
not tell which one is right. The 7 recordings are private voice messages (about 7 s to 130 s of
casual Russian speech with some English technical terms) and are **not published**; only the
timing log and the scripts are. Re-run the comparison on your own recordings.

## Setup

- Recordings: 7 private voice messages (`clip-01` ... `clip-07` in `timing.jsonl`), converted to
  mono 16 kHz mp3 for the cloud models; GigaAM got mono 16 kHz wav.
- Cloud: Gemini chat models with audio input, reached through an API aggregator (kie.ai), same
  prompt for all (a plain "transcribe verbatim, return only text" instruction); one extra variant
  with a stricter "do not add or guess, mark unclear words" prompt.
- Local: GigaAM-v3 RNN-T (sherpa-onnx) on CPU only, 8 vCPU virtual machine, 4 threads
  (`../gigaam_serve.py`). No GPU.
- Runs of different engines overlapped in time on the same machine and network, so wall-clock
  numbers include queueing and are **noisy**: they are indicative, not a latency benchmark.
- Dedicated speech-to-text endpoints (Whisper, vendor "transcribe" models) were not reachable
  through the aggregator or lacked credentials and were **not tested**.

## Results

Distance = mean word-level edit distance to the other engines' transcripts, fillers and
punctuation removed (`analyze.py`); lower = closer to the group. Time = median seconds per
file over successful calls (fastest-slowest in brackets).

| Engine | Succeeded | Distance to consensus | Median time | Notes |
|---|---|---|---|---|
| gemini-3.1-pro | 5/7 | 7.0% | 61 s (39-142) | 2 read timeouts at 180 s; keeps hesitations (11 filler tokens) |
| gemini-3.6-flash | 7/7 | 7.5% | 12 s (6-54) | fastest of the cloud models together with 3.7/3.8; best on Latin-script terms |
| gemini-3.6-flash, strict prompt | 7/7 | 8.0% | 58 s (11-143) | slower, no visible gain from the stricter prompt |
| gemini-3.5-flash | 7/7 | 8.2% | 17 s (8-133) | softens profanity; some terms mis-spelled |
| gemini-3.8-flash | 7/7 | 8.4% | 13 s (7-40) | many filler tokens (59), word fragments kept |
| gemini-3-pro | 7/7 | 8.5% | 44 s (19-55) | most verbatim, most fillers (61) |
| gemini-3.7-flash | 6/7 | 8.6% | 12.5 s (7-90) | 1 timeout on the longest file |
| **GigaAM-v3 (local, CPU)** | 7/7 | 9.4% | 13 s (4-27) | free, offline; worst on Latin-script terms; punctuation present |

Spread across all engines is only 7.0-9.4%: on this material the engines mostly agree, and
differences are in terms, hesitations and small word choices.

## What the texts showed (qualitative, versus consensus - not verified against audio)

- Latin-script technical terms and brand names are where the engines differ most: GigaAM tends to
  write them phonetically in Cyrillic or garble them; the Gemini models are better but
  inconsistent between versions. A post-processing dictionary
  (`../gigaam_words.example.json`, see `../gigaam_serve.py`) fixes the recurring cases for GigaAM.
- One local transcript dropped a short word that all other engines had and appended one extra
  word at the end of the longest recording; the Gemini models occasionally truncated or
  softened words instead. Without a reference these are disagreements, not proven errors.
- Pro-tier models are the most literal but 3-5x slower and the least reliable in this run.
- Cost of the cloud runs was about 0.04 USD for ~55 calls (balance difference reported by the
  aggregator; not independently verified).

## Limits - read before quoting any number

- n = 7 clips, mostly one speaker, no ground truth; a different mix could reorder the engines.
- Distance to consensus rewards agreeing with the majority (which is mostly Gemini variants), so
  it is biased against the one non-Gemini engine.
- Timings come from overlapping runs; model ids and availability on the aggregator change.
- The plain prompt in `run.py` is a paraphrase of the original; the strict prompt is verbatim
  except that product names in its example were generalised.

## Reproduce

```bash
export AGGREGATOR_API_KEY=...            # never commit keys
python run.py gemini-3.6-flash my_clips/*.ogg
python run.py gigaam my_clips/*.ogg      # needs GIGAAM_MODEL_DIR, see ../gigaam_serve.py
python analyze.py raw timing.jsonl
```
