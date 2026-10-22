# Fish Audio (cloud TTS API; open-weights relatives Fish Speech / OpenAudio S1)

- **Kind:** TTS, cloud API (also has voice cloning and a public voice library)
- **Upstream:** https://fish.audio (API docs: https://docs.fish.audio), open code: https://github.com/fishaudio/fish-speech
- **Client scripts in this repo:** `tts/fish/fish_tts.py`, `bench/fish_stream_bench.py`, `bench/fish_timestamps.py`
- **Measured:** September 2026, model `s2.1-pro-free`, Russian text.

## License and terms

| Item | Terms | Source / status |
|---|---|---|
| Hosted API (`api.fish.audio`) | Governed by the vendor's terms of service and pricing page, not by a software license. The vendor's announcement of the free model states commercial use is allowed, with a request to contact them in advance above USD 1M annual revenue, and no latency/availability guarantees on the free tier. | From the vendor announcement as recorded in the project notes (2026-09-15). **Not re-verified** at write time (the pricing page URL tried on 2026-10-22 returned 404). Read the current terms before commercial use. |
| Open weights `fishaudio/openaudio-s1-mini` (Fish Speech / OpenAudio S1 family) | **CC-BY-NC-SA-4.0.** The model card says "This model is permissively licensed under the CC-BY-NC-SA-4.0 license." **NON-COMMERCIAL.** | Verified on the Hugging Face model card, 2026-10-22. |
| `fish-speech` repository (code) | **Fish Audio Community License Agreement, Version 1.0**: research and non-commercial use free; commercial use only while annual revenue is below USD 500,000; offering a hosted service needs a separate written license. Some weights (e.g. S2-Pro) are under a separate "Fish Audio Research License" and are excluded from that agreement. | Read from `LICENSE` on GitHub `main`, 2026-10-22. Summary, not legal advice. |

Commercial use summary: **API = depends on current terms (unverified today); S1 open weights = NON-COMMERCIAL; code = conditional (revenue cap).**

Prices (always with a date): the pricing page showed **USD 15 per 1M UTF-8 bytes** for `s2.1-pro` / `s2-pro` / `s1`, USD 0 for `s2.1-pro-free`, ASR USD 0.36/hour, Voice Design USD 0.01/request -- **as checked 2026-09-15; not re-checked since, prices change.** The free model was announced as free until 2026-11-30.

## Install and run

No local install; API only.

```bash
export FISH_API_KEY=...                       # never pass the key on the command line
python3 tts/fish/fish_tts.py --search --language ru --limit 20
python3 tts/fish/fish_tts.py --voice <reference_id> --text-file phrases.txt \
        --out-dir ./out --format wav --model s2.1-pro-free
python3 bench/fish_stream_bench.py --voice <reference_id> --text "..." --latency low
python3 bench/fish_timestamps.py   --voice <reference_id> --text "..." --out a.mp3
```

The model is chosen by an HTTP **header** `model: s2.1-pro-free`, not a body field. Voices are addressed by `reference_id`. Tags such as `[excited]` are written inline in square brackets.

## Hardware

Cloud service; server hardware unknown. Clients: a Linux container behind a VPN and a Linux host without VPN, both CPU-only (nothing runs locally). The origin server was identified from response headers as being in the US West.

## Measured numbers (`s2.1-pro-free`, Russian)

| Metric | Value | Notes |
|---|---|---|
| Throughput on a 6-utterance dialogue | 35.1 s of audio synthesised in 18.5 s (about 1.9x faster than real time); 1.6-4.1 s per utterance | Other library voices: 12-16 s of audio took 5.8-18.7 s |
| Time to first audio, `latency=normal` | 2.31 s (total 2.35 s) | All 50 chunks arrived within 40 ms: a finished file, not a stream |
| `latency=balanced` | 1.66 s (total 3.09 s) | Real streaming |
| `latency=low` | 1.00-1.06 s (total 2.1-3.1 s) | Three repeats: 1.040 / 1.004 / 1.015 s |
| Time spent in the vendor origin (Server-Timing) | 1.0-8.6 s (container via VPN); 1.1-1.6 s (host without VPN) | Edge/network part: 0.005-0.19 s from both |
| Worst observed origin time | 32.4 s (first audio 32.9 s) | Self-recovered within minutes; concurrency counter showed 0 of 5 |
| Typical spread over time | best 1.00 s; usual 1.7-3.7 s; sometimes 6.3-9.3 s | Same request, different times |
| Warm vs. cold connection | first audio 0.57-1.11 s on a reused connection vs 1.26-2.06 s on a new one | About 0.6 s per new TLS connection |
| Word-timestamp stream (`/v1/tts/stream/with-timestamp`) | first event 1.9-3.2 s | Slower than plain streaming |
| Parallelism | 5 parallel: 5/5 OK, first audio 2.63/2.78/2.87 s (min/avg/max); 10 parallel: 10/10 OK, 1.25/1.43/1.65 s | Free model; roughly 40 requests in total were made |
| Cost per minute of speech | 26.4 UTF-8 bytes per second of audio, 1581 bytes/min, about USD 0.0237/min at USD 15/MB | Computed from one dialogue (924 bytes -> 35.1 s); price as of 2026-09-15 |
| Round trip through an STT engine | 12 of 12 utterances recovered word for word, numbers read correctly | Includes inserted interjections |
| Russian voices in the public library | 1020 (the API total saturates around 1000-1020, so this is a ceiling) | |

Emotion tags (one fixed phrase, pitch measured with `bench/pitchstat.py`):

| Tag | Duration | Median pitch | Spread (st) |
|---|---|---|---|
| none | 4.13 s | 235 Hz | 4.69 |
| `[excited]` | 4.46 s | 276 Hz | 4.32 |
| `[regretful]` | 3.25 s | 229 Hz | 4.53 |
| `[apologetic]` | 3.99 s | 242 Hz | 4.95 |
| `[sad]` | 3.58 s | 200 Hz | 4.73 |
| `[laughs]` | 3.53 s | 178 Hz | 4.06 |
| `[regretful][sighing]` | 3.81 s | 136 Hz | 7.67 |
| `[whispering]` | 3.90 s | 131 Hz | 4.23 |

Liveliness and brightness of the cloud voice versus local voices: see `docs/methodology/liveliness-and-brightness.md`.

## Quality notes for Russian

- Tags work in Russian and measurably change delivery; they did not leak into the audio (the STT round trip contained no tag words). `[whispering]` changes timbre but not loudness (the engine normalises loudness).
- Interjections ("uh", "well") and pauses are pronounced, not dropped: word timing showed a 400 ms vowel preceded by a 320 ms pause (STT engines simply omit such sounds).
- Numbers: the `normalize` option documents number normalisation only for English and Chinese, so write Russian numbers as words to be safe. In the test, an amount written in digits was read correctly (judged via the STT round trip), but that is not a guarantee.
- **Word stress was NOT verified.** No listener model was available; pitch/duration heuristics on isolated words were inconsistent and are not evidence.
- Voice cloning: not evaluated beyond using a library voice that had been cloned from an old archive recording.
- The library's top results are dominated by impersonations of public figures; do not use those for business or any commercial work.

## Known pitfalls

1. Two wallets: platform credits (website) are separate from API credit. Paid models returned `402 Insufficient API credit` while the free model worked and consumed no credits.
2. `latency=normal` looks like streaming but is a finished file in transport chunks; judge by chunk arrival times, not chunk count.
3. An `Accept: */*` header suppresses Cloudflare's `Server-Timing`, making it impossible to split edge vs. origin time.
4. SSE parsing: an event is two lines (`event: message`, then `data: {...}`); a naive "line starts with data:" check silently returns nothing.
5. A stress mark inside a word splits it in the timestamp output.
6. Billing is per UTF-8 byte; Cyrillic letters are 2 bytes, so a per-character estimate is about 2x too low.
7. The free tier has no latency or availability guarantees; provide a fallback engine for interactive use.
8. Measure network baseline separately: an empty request through a VPN cost about 1 s purely from connection setup, which looked like engine latency until the Server-Timing split was used.

## Verdict

Best cloud option measured for lively Russian conversational delivery in this project: tag control and real streaming at about 1 s on `latency=low`. The free tier's time to first audio is erratic (1 to 33 s) and its terms can change, and the cloud voice tested was spectrally dull above 8 kHz (see methodology). Check current terms and pricing before depending on it.
