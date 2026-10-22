"""Russian speech recognition with GigaAM-v3 (RNN-T, punctuation) via sherpa-onnx, CPU-only.

Usage:
    python gigaam_serve.py <audio.wav> [--out file.txt] [--timing] [--json segments.json]
                           [--threads 4] [--chunk 20]

Requirements: `pip install sherpa-onnx numpy`; the model directory (encoder.int8.onnx,
decoder.onnx, joiner.onnx, tokens.txt) downloaded from
https://huggingface.co/csukuangfj/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16
into $GIGAAM_MODEL_DIR (default: ./models/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16).
Input must be mono 16-bit 16 kHz wav (prepare with ffmpeg, see stt_one_shot.py).

--json writes timestamped segments [{"start","end","text"}]; segments are cut at pauses
(signal energy) rather than on a fixed grid, so two channels of a call can be transcribed
separately and merged by time.

Post-processing dictionary: a JSON file {"rules": [["as heard", "as written"], ...]} read from
$GIGAAM_WORDS (default: gigaam_words.example.json next to this script). It repairs terms the
recognizer does not know (brand names, jargon) after decoding.
"""
import argparse, json, os, re, sys, time, wave

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get(
    "GIGAAM_MODEL_DIR",
    os.path.join("models", "sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16"))
WORDS_PATH = os.environ.get("GIGAAM_WORDS", os.path.join(HERE, "gigaam_words.example.json"))


def load_words():
    try:
        with open(WORDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return [(str(a).strip(), str(b).strip())
                for a, b in data.get("rules", []) if str(a).strip()]
    except Exception:
        return []


def apply_words(text, rules):
    """Whole words/phrases, case-insensitive; longer rules first so a two-word rule fires before
    its one-word prefix. Boundaries are letters/digits, so a short rule does not rewrite
    inflected forms of the same word."""
    for spoken, written in sorted(rules, key=lambda r: -len(r[0])):
        if spoken.lower() in text.lower():
            pattern = re.compile(
                r"(?<![а-яёa-z0-9])" + re.escape(spoken.lower()) + r"(?![а-яёa-z0-9])",
                re.IGNORECASE,
            )
            text = pattern.sub(lambda _m: written, text)
    return text


def read_wave(path):
    """The model expects mono 16 kHz; the file must already be prepared (ffmpeg)."""
    with wave.open(path) as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise SystemExit("need wav: mono, 16-bit")
        rate = f.getframerate()
        raw = f.readframes(f.getnframes())
    import numpy as np
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def speech_spans(samples, rate, max_len, min_pause=0.6, pad=0.2):
    """Speech spans split at pauses: 30 ms frames, threshold relative to the recording's own loudness.
    A fully silent channel yields an empty list, not an error."""
    import numpy as np
    frame = int(0.03 * rate)
    n = len(samples) // frame
    if n == 0:
        return []
    rms = np.sqrt((samples[:n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = float(np.percentile(rms, 95))
    if loud < 0.003:
        return []
    voiced = rms > max(0.003, loud * 0.12)
    spans, start, quiet = [], None, 0
    gap = int(min_pause / 0.03)
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap:
                spans.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        spans.append((start, n - quiet))
    out = []
    p = int(pad / 0.03)
    for a, b in spans:
        if b - a < 5:            # shorter than 150 ms: a click, not speech
            continue
        a, b = max(0, a - p) * frame, min(n, b + p) * frame
        step = int(max_len * rate)
        for s in range(a, b, step):  # long monologue: pieces no longer than max_len
            out.append((s, min(b, s + step)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--json", help="write timestamped segments to this file")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=float, default=20.0,
                    help="split the recording into N-second pieces")
    args = ap.parse_args()

    import sherpa_onnx

    t0 = time.time()
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=os.path.join(MODEL_DIR, "encoder.int8.onnx"),
        decoder=os.path.join(MODEL_DIR, "decoder.onnx"),
        joiner=os.path.join(MODEL_DIR, "joiner.onnx"),
        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
        num_threads=args.threads,
        provider="cpu",
        model_type="nemo_transducer",
        debug=False,
    )
    t_load = time.time() - t0

    samples, rate = read_wave(args.audio)
    duration = len(samples) / rate

    t1 = time.time()
    rules = load_words()
    if args.json:
        segs = []
        for a, b in speech_spans(samples, rate, args.chunk):
            s = rec.create_stream()
            s.accept_waveform(rate, samples[a:b])
            rec.decode_stream(s)
            t = s.result.text.strip()
            if t:
                segs.append({"start": round(a / rate, 2), "end": round(b / rate, 2),
                             "text": apply_words(t, rules) if rules else t})
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(segs, f, ensure_ascii=False)
        print(" ".join(x["text"] for x in segs))
        return
    step = int(args.chunk * rate)
    parts = []
    for start in range(0, len(samples), step):
        s = rec.create_stream()
        s.accept_waveform(rate, samples[start:start + step])
        rec.decode_stream(s)
        parts.append(s.result.text.strip())
    text = " ".join(p for p in parts if p).strip()
    if rules:
        text = apply_words(text, rules)
    t_run = time.time() - t1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if args.timing:
        print(f"[GigaAM-v3 | load {t_load:.1f}s | decode {t_run:.1f}s | "
              f"audio {duration:.0f}s | {duration / max(t_run, 0.01):.1f}x faster than real time]",
              file=sys.stderr)
    print(text)


if __name__ == "__main__":
    main()
