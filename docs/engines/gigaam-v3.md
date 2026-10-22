# GigaAM-v3 (SberDevices) - offline Russian ASR, sherpa-onnx export

**Kind:** STT, local (CPU via sherpa-onnx; also runs on GPU elsewhere). **Upstream:**
https://github.com/salute-developers/GigaAM , https://huggingface.co/ai-sage/GigaAM-v3 .
**ONNX export used here:**
https://huggingface.co/csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16
(RNN-T transducer with punctuation, int8 encoder).

GigaAM v3 is a family on a self-supervised (HuBERT-style) encoder pretrained on about 700,000 hours of
Russian speech: `ctc`, `rnnt`, and end-to-end `e2e_ctc` / `e2e_rnnt` variants that output
punctuated, normalised text. The sherpa-onnx export used here is the punctuated RNN-T one.
It is an **offline** model: no streaming, text appears only after the whole file/segment is decoded.

## License (checked against the official pages)

| Part | License | Source |
|---|---|---|
| Code, `salute-developers/GigaAM` | MIT License, "Copyright (c) 2024 GigaChat Team" | `LICENSE` in the GitHub repo |
| Weights, `ai-sage/GigaAM-v3` | MIT (HF metadata `license: mit`; card body: "License: MIT") | HF model page |
| sherpa-onnx export repo (`csukuangfj/...`) | no license field in HF metadata; its README says to see the GigaAM LICENSE above; the repo carries an MIT `LICENSE` copy | HF repo |

Commercial use: **allowed** (MIT). Caveat: neither the GitHub README nor the HF card states a
separate weights license; MIT is stated for the project and the model card. The export script
folder in sherpa-onnx is Apache-2.0 / Xiaomi-copyrighted; it is not included in this repo.
The weights are not included either.

## Install and run (as measured)

```bash
pip install sherpa-onnx numpy          # plus ffmpeg on PATH
mkdir -p models && cd models
# download from the HF export repo above: encoder.int8.onnx decoder.onnx joiner.onnx tokens.txt
# (total repo storage is about 232 MB)
cd ..
export GIGAAM_MODEL_DIR=./models/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16
python stt/gigaam_serve.py clip.wav --timing        # mono 16 kHz 16-bit wav
python stt/stt_one_shot.py clip.ogg                 # any format, runs ffmpeg first
```
Loader: `sherpa_onnx.OfflineRecognizer.from_transducer(..., model_type="nemo_transducer",
provider="cpu", num_threads=4)`. Long audio is cut into 20 s pieces (or at pauses with `--json`).
`stt/gigaam_serve.py` also applies an optional replacement dictionary
(`GIGAAM_WORDS`, example in `stt/gigaam_words.example.json`).

## Hardware used

- Cloud-comparison runs: 8 vCPU / 24 GB virtual machine, **CPU only**, 4 threads, other jobs running
  on the same box.
- A separate desktop PC was also used by the author and gave about 40x real time; which device
  (CPU or GPU) ran it was not recorded, so that figure is indicative only.

## Measured numbers

- 7 short voice messages (about 7-130 s each) on the VM: median 12.8 s per file, range 3.8-26.6 s,
  with concurrent load (`../../stt/cloud-compare/timing.jsonl`). Source-code comment estimate for
  long calls: about 6x real time on CPU (an estimate, not a benchmark).
- Consistency with other engines on those clips: mean word distance to the others 9.4%, the highest
  of the 8 engines compared, while all engines lie within 7.0-9.4% (see
  `../../stt/cloud-compare/SUMMARY.md`). Not a WER.
- Vendor-published WER (GigaAM-v3 model card, not measured here): RNN-T average 8.4% over their
  five test domains (open datasets 2.6, Golos far-field 3.9, natural speech 6.9, disordered speech
  19.2, call-centre 9.5); CTC average 9.2%; Whisper 25.1% on the same table.
- Model memory: about 160 MB (author's estimate in the T-one comparison notes).

## Quality notes for Russian

- Punctuation and capitalisation come from the model; digits and Latin script are possible
  (unlike T-one), but Latin technical terms and brand names are usually written phonetically in
  Cyrillic or garbled ("Hap" / "hape" for a product name). A post-processing dictionary
  repairs recurring cases.
- Dropped a short phrase that all other engines kept in one clip and appended one extra word at the
  end of the longest clip (disagreement with the group; not verified against audio).
- Does not invent words or soften profanity, unlike some LLM-based transcribers.
- Noisy two-speaker audio: handled clearly better than T-one on the same clip (author's notes).

## Pitfalls

- Input must be mono 16-bit 16 kHz wav; convert with ffmpeg first.
- Silent channels: `speech_spans` returns an empty list instead of failing.
- Stereo call recordings: split the channels, transcribe separately, merge by time
  (`stt_one_shot.py --dialog`).
- No streaming; for live phone dialogue use a streaming engine (T-one) and re-read afterwards.

## Verdict

Best accuracy among the local engines here, free and offline, but offline-only and weak on Latin
terms without a dictionary. Good default for batch transcription of recordings.
