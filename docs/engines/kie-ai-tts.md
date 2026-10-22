# kie.ai TTS (aggregator of cloud TTS models: Google Gemini TTS, ElevenLabs)

- **Kind:** TTS, cloud aggregator (one API key and one credit balance for many third-party models)
- **Upstream:** https://kie.ai, docs: https://docs.kie.ai
- **Client script in this repo:** `tts/kie/kie_tts.py`
- **Measured:** 2026-09-15, Russian text. Model availability on an aggregator changes quickly; re-check before relying on any line below.

## License and terms

kie.ai is a hosted service; there are no weights or code to license. Terms of use and the licenses of the underlying models (Google Gemini TTS, ElevenLabs) are set by kie.ai and the respective vendors.

- Commercial use: **unverified.** The kie.ai docs landing page (fetched 2026-10-22) lists image, video, music and chat models and refers to the Market page for the current model list; it did not state TTS models or commercial-use terms. Read the terms of service and each model's page before commercial use.
- The ElevenLabs endpoints were not functional when tested (below), so their terms on this aggregator are moot for the measurements here.

## Models seen in the documentation (sitemap, 2026-09-15)

| Model id | What it is | Status when tested |
|---|---|---|
| `elevenlabs/text-to-speech-multilingual-v2` | ElevenLabs Multilingual v2 | failed: `500` |
| `elevenlabs/text-to-speech-turbo-2-5` | ElevenLabs Turbo v2.5 | failed: `500` |
| `elevenlabs/text-to-dialogue-v3` | ElevenLabs v3 dialogue (audio tags) | failed: `500` |
| `elevenlabs/audio-isolation` | noise removal | failed: `500` |
| `google/gemini-2-5-pro-tts` | Gemini 2.5 Pro TTS | worked; cheaper |
| `google/gemini-3-1-flash-tts` | Gemini 3.1 Flash TTS | worked; about 2x the price |

Not offered on kie.ai at the time: ElevenLabs Flash v2.5, MiniMax Speech, Qwen-TTS, OpenAI TTS, voice cloning, streaming audio output.

## Install and run

API only, no local install.

```bash
export KIE_API_KEY=...          # never on the command line
python3 tts/kie/kie_tts.py --model google/gemini-2-5-pro-tts --input @input.json --out a.wav
```

Flow: `POST https://api.kie.ai/api/v1/jobs/createTask` -> poll `GET .../jobs/recordInfo?taskId=...` -> download the result URL. Example `input.json`:

```json
{"temperature": 1,
 "scene": "A phone call to a customer from a call-centre agent.",
 "sample_context": "Lively conversational Russian, like a person, not a newsreader.",
 "speakers": [{"speaker_id": "Speaker 1", "voice_name": "Charon",
               "audio_profile": "Call-centre agent, speaks lively",
               "accent": "Neutral", "style": "Vocal Smile", "pace": "Natural"}],
 "dialogue_turns": [{"speaker_id": "Speaker 1", "text": "Здравствуйте! Вам удобно сейчас говорить?"}]}
```

Controls: `scene` and `sample_context` (free text; Russian is understood), `style` (e.g. Vocal Smile, Newscaster, Whisper, Empathetic, Promo/Hype, Deadpan), `pace` (Natural, Rapid Fire, The Drift, Staccato), inline tags such as a pause marker. 30 preset voices in total; 13 were rendered (10 male, 3 female).

## Hardware

Cloud service; server hardware unknown. Client: a Linux container behind a VPN, CPU-only.

## Measured numbers (2026-09-15)

Billing is by duration of the finished audio, consistent over 25 measurements. 1 credit is about USD 0.005 (28.73 credits = USD 0.144 spent in total).

| | Gemini 2.5 Pro TTS | Gemini 3.1 Flash TTS |
|---|---|---|
| Credits per second of audio | 0.071 | 0.142 |
| Per 1000 characters | 4.9 credits = USD 0.025 | 9.3 credits = USD 0.047 |
| Per minute of continuous speech | USD 0.021 | USD 0.043 |
| Speech rate | about 14-15 characters/s | about 14-15 characters/s |
| Real-time factor (generation time / audio time) | 1.2 | 0.66 |
| Delay before the file is ready | 8-22 s | 6-12 s |
| Output | WAV 24 kHz mono PCM16 | same |

- Downloading the result took an additional 5-6 s through the VPN used.
- ElevenLabs endpoints: 11 calls over several sessions, all `state: fail, failCode: 500, failMsg: "Internal Error, Please try again later."`, `costTime` 93-106 ms, `creditsConsumed: 0.0` (nothing billed). Failure came before any synthesis could start, which points to the integration on the aggregator side.
- Random internal task failures on the working models: 2 of 27 tasks; retry (failures are not billed).
- STT round trip: the text was recovered word for word (acronyms, a city name, an amount in words), so speech is intelligible. This says nothing about stress.

## Quality notes for Russian

- Gemini TTS pronounces Russian intelligibly (round trip above). **Word stress and liveliness were not verified**: feeding recordings back to an LLM judge returned "no audio received" in 3 of 4 tries and an invented transcript in the fourth, so no judgement is reported.
- Style, pace and scene prompts steer delivery; voices are fixed presets, no cloning.
- No streaming: every utterance incurs the full task latency (6-22 s plus download). Fit for pre-rendered prompts, not for live conversation.

## Known pitfalls

1. Gemini TTS voiced **only the first entry of `dialogue_turns`**; later turns were dropped silently (3 turns gave a 5.6 s file instead of 18 s). Put all text in one turn.
2. ElevenLabs Multilingual v2 rejects `language_code` (only Turbo/Flash 2.5 accept it).
3. The ElevenLabs voice field is named `voice`, although the validator error message says `voiceId cannot be empty`.
4. The result URL returns `403` without a browser-like `User-Agent`; the task is already billed at that point.
5. Retry random failures; failed tasks cost nothing.
6. Aggregator availability drifts: ElevenLabs endpoints were down on the day of testing; re-test periodically (failed calls are free).

## Verdict

Cheap (about USD 0.02/min of continuous speech on the Pro model) and intelligible, but no streaming and no cloning, and the ElevenLabs endpoints did not work during testing. Use it only for pre-rendered phrases; for live dialogue use a streaming engine. Prices as of 2026-09-15 and subject to change.
