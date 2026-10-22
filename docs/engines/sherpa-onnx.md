# sherpa-onnx - local inference runtime used for GigaAM

**Kind:** STT runtime (also TTS, VAD, diarization); not a model. **Upstream:**
https://github.com/k2-fsa/sherpa-onnx (next-gen Kaldi + ONNX Runtime).

Runs speech models fully offline on many platforms and from many languages (Python API used here).
In this project it executes the GigaAM-v3 RNN-T export; see `gigaam-v3.md`.

## License (checked)

Apache-2.0 (GitHub repository license; verified on the repo page). Commercial use: **allowed**.
Model files run by it carry their own licenses (GigaAM export: MIT, see `gigaam-v3.md`).

## Install and run

```bash
pip install sherpa-onnx numpy
```
```python
import sherpa_onnx
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="encoder.int8.onnx", decoder="decoder.onnx", joiner="joiner.onnx",
    tokens="tokens.txt", num_threads=4, provider="cpu", model_type="nemo_transducer")
s = rec.create_stream(); s.accept_waveform(16000, samples)   # float32 mono in [-1, 1]
rec.decode_stream(s); print(s.result.text)
```
Full script: `../../stt/gigaam_serve.py`.

## Hardware used

8 vCPU / 24 GB virtual machine, CPU only (provider `cpu`), 4 threads.

## Measured numbers

None of its own: speed is that of the model run through it (GigaAM: median 12.8 s per file on
files of about 7-130 s under concurrent load; see `gigaam-v3.md`).

## Notes

- The sherpa-onnx repository also contains the export scripts for the model; the exported ONNX files
  (about 232 MB total for this model) are downloaded from Hugging Face, not stored here.
- Input must match the model's feature extractor (mono 16 kHz here).

## Verdict

A clean way to run GigaAM on CPU with a small Python dependency (no torch). Apache-2.0.
