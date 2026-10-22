# Measuring "liveliness" and "brightness" of synthetic speech

Two cheap, objective proxies used to compare TTS voices (local vs. cloud) without arguing about taste. They complement, and never replace, listening.

| Proxy | What it captures | Tool |
|---|---|---|
| Liveliness | How much the pitch moves (monotone announcer vs. lively speaker) | `bench/pitchstat.py` |
| Brightness | How much real high-frequency content there is (wideband vs. dull/16 kHz-sourced) | `bench/bandcheck.py` |

## 1. Liveliness: pitch spread in semitones

`bench/pitchstat.py` estimates F0 by autocorrelation (128 ms frames, 32 ms hop, audio resampled to 8 kHz mono with ffmpeg, search range 70-400 Hz, silence dropped by an energy gate at 2% of peak energy, voiced decision at a normalised autocorrelation above 0.35). It converts voiced F0 to semitones (12 * log2 F0) and reports, per file:

- **sd** -- standard deviation of voiced pitch in semitones ("spread"),
- **range** -- 90th minus 10th percentile in semitones,
- median F0 in Hz, and the voiced fraction.

Semitones are used because that is how the ear perceives pitch differences. Pure-Python, no signal-processing dependency; slow but adequate for short clips.

```bash
python3 bench/pitchstat.py --group "cloud=out/cloud/*.wav" --group "local=out/local/*.wav"
```

Limits: it measures pitch movement, not pleasantness. It does not detect hoarseness, breathiness, rhythm problems or synthesis artefacts. Higher spread is not automatically "better".

### Findings that came out of using it

1. **Conversational speech is livelier than read speech.** Source recordings measured before any synthesis: read speech (crowd-sourced reading corpus) 2.48 st spread / 5.99 st range; spontaneous conversational speech 3.83 st / 8.57 st -- about 55% more spread.
2. **A zero-shot cloning engine transfers the reference's liveliness but compresses it.** Across six voices, sample spread vs. synthesis spread correlated at r = +0.72 (range: r = +0.67). The synthesis kept roughly 80% of the spread and cut the pitch range by more than half (e.g. 14.98 -> 7.03 st for the most expressive female reference).
3. **Select references by liveliness, not by cleanliness.** The two cleanest references (sample spread 2.82 and 2.84 st) produced the deadest voices (2.31 and 2.55 st), worse than the announcer-style baseline (2.81 st). References with sample spread of about 4.5 st or more gave 3.8-4.0 st or better. Rule of thumb adopted: take clips with spread of at least about 4.5 st; cleanliness is the second criterion.
4. **Engine knobs did not help.** On the most expressive female reference: default 4.06 st; stronger reference following (cfg 3) 3.77, (cfg 4) 3.27; double the solver steps (nfe 64) 3.56; sway -0.5 3.83, sway 0 4.01; slower speech (0.92) 4.03; combination 3.30. The defaults were best. A "podcaster" variant of the model was no livelier (3.12 vs 3.05 on plain text; 2.91 vs 3.33 with fillers).
5. **Inserted fillers ("uh...", "well", pauses written into the text) raise spread for some voices only.** Mean over a first set: 3.05 -> 3.33 (+9%); on a conversational reference they lengthened as intended. On a read-style reference they had no effect (3.06 -> 2.98). Per voice the effect ranged from +1.26 st to -1.04 st, so the "with/without fillers" pair must be chosen per voice by listening.

## 2. Brightness: the "gap" above 8 kHz

`bench/bandcheck.py` resamples to 44.1 kHz, applies a steep high-pass at 8 kHz (two 2-pole filters) and compares the RMS level of the filtered signal with the RMS level of the whole file using ffmpeg `astats`. The **gap** = whole-file level minus above-8-kHz level, in dB. **Smaller gap = brighter, wider sound.** It tells a genuine wideband recording from a narrowband one that was merely upsampled (e.g. a 16 kHz recording has almost nothing above 8 kHz).

```bash
python3 bench/bandcheck.py sample1.wav sample2.wav
```

Reference points on this project's material:

| Gap | Meaning |
|---|---|
| 12-20 dB | studio-quality wideband recording, bright |
| about 23 dB | announcer-style sample, fine |
| 30 dB and more | dull (16 kHz source); judged "muffled" by listeners |

Measured sources: 16 kHz spontaneous-speech corpus clips 29.5-32.5 dB (the content above 8 kHz physically does not exist); an announcer sample (24 kHz) 23.2 dB; an acted-speech corpus at 44.1 kHz 12.4-24.9 dB.

Ceiling effect: the local engine outputs 24 kHz audio, so nothing above 12 kHz can be produced. A reference with a 15.2 dB gap came out at 18.7 dB, i.e. the engine "eats" about 3 dB. Cleaning references further, or running audio restoration on them, is pointless once the engine is the limit.

## 3. Local vs. cloud: comparison table

Same six-utterance call script, same text, different engines. "Local" is an open-weights Russian flow-matching TTS (ESpeech) cloned from references of an acted-speech corpus (licence in section 5), running on a GPU desktop; it synthesised at about 3-5x real time (RTF 0.20-0.50) with speech starting after about 1.5 s. "Cloud" is Fish Audio `s2.1-pro-free` with one cloned library voice.

| Voice | Liveliness (sd, st) | Brightness (gap above 8 kHz) |
|---|---|---|
| Cloud voice (Fish, one library voice) | 4.71 | dull, 43 dB gap |
| Local male voice `e-m3`, with filler insertions | 4.53 | bright, 19 dB |
| Local male voice `e-m3`, plain text | 4.22 | 19 dB |
| Local female voice `e-f1` | 3.39 | 19 dB |
| Announcer-style baseline voice | 2.81 | 24 dB |

Reading: the local voice was within about 4% of the cloud voice in liveliness and clearly brighter. Band-level cross-check on the cloud file (44.1 kHz delivery): level in the 8-12 kHz band about -70 dB for the cloud voice vs about -49 dB for the local one; 12-16 kHz about -90 dB for the cloud voice, i.e. essentially empty. The likely cause (a version, not tested) is that the cloned voice came from an old archive recording with no high-frequency content. For telephone use this makes no practical difference, since the line removes those frequencies anyway.

Pitch level and spread of different engines on the same text (cloud voice chosen to be high-pitched; compare spread, not pitch):

| Engine | Median F0 | Spread (st) | Range (st) |
|---|---|---|---|
| Fish, plain | 236.5 Hz | 4.71 | 10.79 |
| Fish, conversational text | 248.0 Hz | 4.71 | 11.15 |
| ESpeech, conversational | 118.8 Hz | 2.14 | 5.28 |
| Qwen3-TTS | 128.8 Hz | 2.82 | 7.06 |

The two tables use different phrase sets (call script vs. reference-selection phrases), which is why `e-m3` with fillers is 4.53 in one and 5.03 in the other. Results of the later reference-selection rounds (spread in st / range in st / gap in dB, means):

| Set | Spread | Range | Gap |
|---|---|---|---|
| announcer-style clips (baseline) | 2.81 | 6.27 | 23.5 |
| spontaneous 16 kHz corpus references: lively but dull | 4.23 | 8.00 | 34.6 |
| acted corpus, six voices | 3.24 | 7.55 | 23.7 |
| acted `e-f1` | 3.86 | 8.71 | 18.7 |
| acted `e-m3` | 4.10 | 9.95 | 18.9 |
| acted `e-m3` with fillers | 5.03 | 11.03 | 19.9 |

### Caveats

- The proxies measure intonation and bandwidth only. Listening is mandatory.
- The comparison is against **one specific cloud voice**, not against the cloud service in general; other voices will score differently.
- Fillers help only some voices (see above).
- Single speaker per bar, few utterances: no confidence intervals were computed.

## 4. Why the cloud/local question matters

A cloud free tier can disappear or become paid (announced end date at the time of testing: 2026-11-30), while a local engine costs nothing per minute and works offline. Keeping a local branch measured as roughly equal in liveliness and better in brightness is a cheap insurance policy.

## 5. Where to get legal reference voices

Cloning needs a reference clip (about 5-12 s of clean speech with an exact transcript; for the engine used here at most about 12 s). Use only recordings you have the rights to, or datasets that explicitly allow it. Never use voices from video platforms, podcasts, films or recognisable public figures -- that is someone's voice without consent. If you record your own, put at least 10 s of clean speech you hold the rights to into `./refs/`.

Licences below are as recorded in the project notes (2026-09/10); the two marked "checked" were re-read on the dataset card on 2026-10-22. **Always re-check the current card before use.**

| Dataset | Licence | Fit for commercial reference use? |
|---|---|---|
| `Aniemore/resd_annotated` (about 1.4k acted improvised clips by 20 actors; part is 44.1 kHz) | MIT (checked) | Yes; the best source found: lively and wideband. Selection used: 44.1 kHz only, at most 12 s, calm/friendly emotions, then liveliness and cleanliness; 20 usable clips out of 1396 (13 female, 7 male) |
| `bond005/sova_rudevices` (SOVA RuDevices, spontaneous speech, 16 kHz) | CC-BY-4.0 (checked) | Yes with attribution; lively but dull (no content above 8 kHz) |
| Common Voice 17 (`fsicoli/common_voice_17_0` mirror) | CC0-1.0 per notes (the official dataset page showed no license field when fetched; it now points to Mozilla Data Collective) | Legally clean, but read speech: use only as a control group |
| Golos (Sber) | custom "public license with attribution and conditions reserved" | Not used: not a standard open licence, needs separate legal review; also short commands rather than conversation |
| Silero open_stt | CC-BY-NC-4.0 per notes | No: non-commercial |
| Dusha (Sber, emotions) | no licence stated | Not used: unknown means unusable |
| ESpeech podcasts dataset | CC-BY-NC per notes | No: non-commercial |
| Datasets of identifiable streamers' voices | various (e.g. Apache-2.0) | Not used: a dataset licence is not the speaker's consent to commercial voice cloning |
| TTS-generated datasets (named after OpenAI / ElevenLabs voices) | n/a | Not used: synthetic, not natural speech |
| Video-platform scrapes | n/a | No: voices without consent |
| `TeraTTS/betterset` | n/a | Gated (HTTP 401) |

Practical tips used while collecting: fetch clips one at a time through the Hugging Face datasets-server API (`bench/pick_live_refs.py`) instead of downloading whole corpora, and range-request only the first megabytes of a large archive. Verify that the dataset transcript matches the audio: the engine consumes both, and a wrong transcript degrades the clone. Do not cut a reference clip shorter to fit the limit if that breaks the audio/text match; choose a shorter clip instead.

Reproduce: `python3 bench/pick_live_refs.py 1200 6 ./live_pick`, then `python3 bench/pitchstat.py --group "refs=./live_pick/*.wav"` and `python3 bench/bandcheck.py ./live_pick/*.wav`.
