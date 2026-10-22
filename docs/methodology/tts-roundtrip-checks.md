# TTS checks without a listening panel: ASR round-trip and speaker similarity

Two objective checks used for Chatterbox (and applicable to any TTS):

## 1. Intelligibility by ASR round-trip

1. Synthesize a sentence.
2. Transcribe the audio with an independent ASR (here: Gemini 2.5 Flash through a transcription helper).
3. Compare the transcript with the input text. Whatever comes back verbatim was pronounced intelligibly.

Use a **paired design** for text-normalization questions: the same sentence with digits and with spelled-out
numbers. For Chatterbox, digits came back as nonsense while the spelled-out version round-tripped perfectly
(including back into "4521" and "15 750 ₽"), which isolates the cause to number handling, not to the voice.
Caveat: the ASR can also be wrong, so a failure is a prompt to inspect, not proof by itself.

## 2. Voice-cloning similarity by speaker-embedding cosine

1. Pick a clean reference. Instead of choosing "by eye", run a VAD and concatenate consecutive speech segments
   without long pauses (here: a 12.3 s excerpt with 12.0 s, 98 %, of speech).
2. Embed the reference, the clone outputs and unrelated voices with the same speaker encoder
   (`tts/chatterbox/similarity.py`, Chatterbox `VoiceEncoder`) and compute cosine similarity.
3. Always include two anchors: a **ceiling** (another clip of the same real speaker; 0.887 in our run) and a
   **floor** (an unrelated voice; 0.70-0.74). Read the clone's score as its position between them
   (clone 0.80-0.82 = about 60 % of the way).

Caveat: if the metric's encoder is the same one the model conditions on, the score flatters the model.
It does not replace a listening test. Use references you have the rights to use; none are shipped here.
