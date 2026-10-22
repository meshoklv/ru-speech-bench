# Kokoro-82M (TTS)

**Kind:** TTS, local (CPU or GPU, ONNX) | **Upstream:** https://huggingface.co/hexgrad/Kokoro-82M ,
https://github.com/hexgrad/kokoro | **ONNX runtime wrapper:** https://github.com/thewh1teagle/kokoro-onnx |
**Russian fine-tune:** https://huggingface.co/zaakirio/kokoro-ru

## What it is

An 82M-parameter StyleTTS2-style TTS that consumes IPA phonemes. v1.0 ships 54 voices in 8 languages;
**Russian is not an official language** (not re-verified on the model page; the HF page does not list the
languages). Two ways to get Russian speech were tested:

1. **Stock model + espeak-ng `ru` phonemes.** All 128 model-vocabulary symbols were covered (0 phonemes
   dropped, checked with the tokenizer), so Russian is voiced with stock (English/other) speaker styles.
2. **`zaakirio/kokoro-ru`**, a Russian fine-tune (29.3 h of Russian speech, 16 speakers; voices `sveta`,
   `masha` on a shared checkpoint, `dima` on a separate one).

## License of code and weights (checked 2026-10)

| Part | License | Commercial use | Source |
|---|---|---|---|
| Kokoro-82M weights | **Apache-2.0** (HF metadata `apache-2.0`) | yes | huggingface.co/hexgrad/Kokoro-82M |
| `kokoro` code | Apache-2.0 | yes | github.com/hexgrad/kokoro |
| `kokoro-onnx` package | **MIT** | yes | github.com/thewh1teagle/kokoro-onnx |
| `kokoro-ru` weights | **OpenRAIL** (HF metadata tag `openrail`; README: "Weights are OpenRAIL, code is Apache-2.0") | unverified: OpenRAIL normally allows commercial use *with use restrictions*; the exact terms were not shown on the page | huggingface.co/zaakirio/kokoro-ru |

The `kokoro-ru` card says its voices come from a "Dialogs corpus" described as "studio, consented actors,
OpenRAIL". We did not audit that corpus.

## Install and run

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install kokoro-onnx soundfile     # 0.6.1 in our run; bundles its own espeak-ng (espeakng_loader)
```

* Stock model files: `kokoro-v1.0.onnx` (310 MB, float32) and `voices.bin` (27 MB, 54 voices) from the
  `kokoro-onnx` GitHub release assets.
* Do **not** install a system espeak-ng alongside: the model was trained with the bundled espeak data 1.51;
  phonemes changed in 1.52.
* `kokoro-ru`: ONNX exports (`ru-base.onnx`, `ru-dima.onnx`) plus float32 style packs `sveta.bin`,
  `masha.bin`, `dima.bin` of shape `[510, 1, 256]`, passed as the `voice=` array with `lang="ru"`.
* HTTP wrapper with CPU/CUDA selection and warm-up: `tts/kokoro/pc-server/server.py`
  (listens on 127.0.0.1 only; no authentication). GPU check: `tts/kokoro/pc-server/gpu_probe.py`.
* GPU: `onnxruntime-gpu` 1.22 worked with CUDA 12 (1.30 needs CUDA 13); the CUDA session must be created
  with a memory cap (`gpu_mem_limit`), and the server falls back to CPU if the CUDA provider does not load.

## Hardware used for the measurements

* **Container CPU:** 8 vCPU, shared, load average ~4 (the CPU model is not recorded in this document; other
  measurements on the same host used Xeon E5-2695 v2 @ 2.4 GHz, so the same CPU is *likely*, not verified).
* **Desktop CPU:** AMD Ryzen 9 5950X, 16 threads (CPU-only runs).
* **Desktop GPU:** NVIDIA RTX 3090 Ti via onnxruntime CUDA EP (3 sessions used ~1.9 GB VRAM).

## Measured numbers

Source: console timings from the benchmark sessions (September 2026); no JSON results file was saved.

| Config | Short phrase (3.3 s audio) | Long phrase (8.8 s audio) |
|---|---|---|
| float32, intra_op=2 | RTF 1.49 | RTF 1.40 |
| float32, intra_op=4 | RTF 0.86 | RTF 0.81 |
| float32, intra_op=8 | **RTF 0.77** (~2.6 s) | RTF 0.76 (~6.7 s) |
| int8, intra_op=8 | RTF 3.38 | RTF 2.99 |

* **int8 dynamic quantization was ~4x slower** than float32 on that CPU.
* Streaming (`create_stream`) did not help: a 8.8 s phrase came back as ONE chunk after 6.4 s (the splitter
  does not cut short phrases); split by sentence yourself.
* Model load is lazy: the first call costs ~3 s extra. Warm second call: 3.15 s for a ~4 s reply.
* Desktop CPU (5950X, 16 threads): a 3.3 s reply in **0.66-0.75 s**; long phrase (~7 s audio) ~1.2 s for
  `dima`/`sveta`/`masha`, ~1.4 s for stock `am_adam`.
* Desktop GPU (3090 Ti, CUDA EP): a 5.7 s reply in **0.30 s (RTF 0.05)** vs 0.9 s on the same machine's CPU.
* A server using this model on a remote machine added ~1.5 s of network chain; keep-alive HTTP removed ~1.1 s
  of connection setup. Those are deployment numbers, not model numbers.
* ASR round-trip WER, **reported by the `kokoro-ru` authors, not reproduced here**: Whisper large-v3 on 79
  words, 2.50 % (kokoro-ru) vs 4.38 % (Piper, mean of four voices); on a 55-word stress-heavy set, ~27 % WER
  with espeak G2P vs ~22 % with their RUAccent G2P. The same card notes Piper `irina` transcribed perfectly
  where kokoro-ru did not.

## Quality notes for Russian

* Intelligible with lively intonation; **stress is the weak point**: espeak places it, sometimes wrongly.
* Numbers: espeak-ru expands them ("пятнадцать дней"); a text normalizer on top did not double-expand.
* Latin script is read letter by letter in English; `3-5 дней` -> "три пять дней"
  (the "до" is lost). Pass identifiers like tracking codes through a normalizer first.
* An explicit stress mark ("за́мок") is **ignored** by espeak-ru, so homographs cannot be controlled. The
  `kokoro-ru` G2P front end (RUAccent) is the upgrade path.
* No voice cloning in this stack: you pick or blend style vectors.

## Known pitfalls

* Concurrent `create()` from threads corrupted output (upstream issue #191); 0.6.1 has an espeak lock, and
  the server adds a global lock.
* A server process with a CPU venv kept answering after the task was repointed at the GPU venv: expose a
  version marker in `/health` and check it.
* Routing prefixes in front-ends must cover the Russian fine-tune voices or requests fall through to another
  engine.
* numpy 2.x may conflict with other packages (e.g. `pyctcdecode` wanted <2); verify imports after install.

## Verdict

Very light and fast (RTF ~0.05 on a GPU, <1 on CPU with 8-16 threads), permissive license on the stock model,
but Russian rides on espeak phonemes: stress errors and no cloning. Use `kokoro-ru` if you accept OpenRAIL terms.
