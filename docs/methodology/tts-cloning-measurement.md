# Measuring voice-cloning TTS without listening panels

Techniques used for the Qwen3-TTS / ESpeech / XTTS-v2 / F5-TTS comparison. None of them replaces listening:
they do not measure intonation, accent, pauses or "liveliness". Always listen to the files before choosing an engine.

## 1. Speed
- **RTF = synth_time / audio_time** (lower is faster; < 1 means faster than real time). Some older scripts here print the inverse
  (`audio/synth`, higher = faster); the docstring of each script says which.
- Wrap the call in `torch.cuda.synchronize()` before reading the clock; report cold run separately from warm runs (median of warm).
- Repeat on an idle GPU; background LLMs change nothing for a compute-bound model, but change free VRAM.

## 2. VRAM
- `torch.cuda.max_memory_allocated()` is the tensor peak, not the occupied card. Also read `nvidia-smi` before/after loading and after
  synthesis (on Windows/WDDM per-process memory shows `[N/A]`, so use the delta of total used memory).
- Whether two models fit together is arithmetic from separate measurements unless you actually run both.

## 3. Latency to first audio
- Streaming APIs differ: check whether the generator yields while generating, or slices a finished waveform.
  Measure time to first yielded chunk and compare with total time; if they are equal, nothing is streamed.
- For non-streaming engines the floor is the synth time of the shortest phrase.

## 4. Intelligibility (round trip)
- Synthesize -> transcribe with an STT engine -> compare with the source text after normalisation (lower-case, no punctuation, e/yo folded).
- STT writes numerals as digits: compute WER **excluding numerals** or write numbers in words in the test phrases and expect digit output.
- Brand names get Latinised ("Neway") and "номер" becomes "No": artefacts of STT shared by all engines; discount them.
- STT sometimes returns its own prompt text on very short clips: re-run before blaming the synthesizer.
- A transcript that starts with extra words is the signature of **reference leakage** (see below).

## 5. Pitch fidelity of cloning
- Median fundamental frequency by frame autocorrelation (40 ms frames, silence dropped). Compare the clone with the reference.
  Measures pitch only (male vs female, "younger" voice), not timbre.

## 6. Speaker similarity
- Cosine similarity of speaker-encoder embeddings (e.g. XTTS's H/ASP encoder) between reference and synthesis. Include a
  **control** (a different voice) to show the metric separates "same" from "other". A home-field encoder may favour its own engine.

## 7. Naturalness prediction
- UTMOS (`tarepan/SpeechMOS`, `utmos22_strong`) is English-trained: use for relative comparison only. It rewards cleanliness
  and can rank a flat but clean voice above a lively one; also score the references themselves.

## 8. Reference-clip hygiene (matters more than the engine)
- Dense speech without pauses, exact transcript, end on a finished sentence (an unfinished tail invites the model to finish it),
  matching the target delivery (calm and friendly for a call-center voice), sample rate >= 24 kHz.
- Engines that infer output length from reference length (F5 family) are stretched by pauses in the reference.
- Check the clip by transcribing/listening, not by its file name.
- Use only audio you have the right to use.

## 9. Reference leakage and seeds
- Keep one cached clone prompt per speaker and identical sampling parameters; vary only the seed.
- Generate several seeds, run the round-trip check, keep the clean take; trimming the stray leading word at a pause (without re-synthesis)
  keeps the voice identical. This is an offline workaround; it does not work in a live call.
