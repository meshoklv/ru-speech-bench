"""Measure ESpeech "streaming" output: time until the FIRST audio chunk arrives.

Usage:
  python esp_stream_bench.py --voice voiceA [--refs ./refs/refs.json] [--runs 4]

F5-TTS's infer_batch_process(streaming=True) yields chunks, BUT it synthesizes each text
segment completely first and only then slices the finished waveform into chunks. So with
one segment the first chunk arrives when everything is done; splitting the text into short
phrases lowers time-to-first-chunk at the cost of total time. Run 0 is cold and excluded
from medians. Needs: GPU, `pip install f5-tts ruaccent soundfile`, ESpeech weights.
RTF = synth_time / audio_time (lower is faster).
"""
import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "espeech"))

import torch  # noqa: E402
import common  # noqa: E402

common.patch_audio_io()

TEXT = "Здравствуйте! Меня зовут Кирилл, барбершоп «Ромашка». Вам удобно сейчас говорить?"
PARTS = ["Здравствуйте!", "Меня зовут Кирилл, барбершоп «Ромашка».", "Вам удобно сейчас говорить?"]
CHUNK = 2048


def stream_runs(label, ra, rt, texts, model, voc, runs):
    from f5_tts.infer.utils_infer import infer_batch_process
    firsts, totals = [], []
    for run in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        first, nch, nsamp, sr = None, 0, 0, 24000
        for item in infer_batch_process(ra, rt, texts, model, voc, streaming=True,
                                        chunk_size=CHUNK, nfe_step=32, progress=None, device="cuda"):
            chunk = item[0]
            if len(item) > 1 and isinstance(item[1], int):
                sr = item[1]
            if chunk is None:
                continue
            if first is None:
                first = time.time() - t0
            nch += 1
            nsamp += len(chunk)
        tot = time.time() - t0
        dur = nsamp / sr
        tag = "cold" if run == 0 else f"run{run}"
        print(f"{label} {tag}: first_chunk={first*1000:.0f} ms  total={tot:.2f}s  audio={dur:.2f}s "
              f"chunks={nch} chunk_len={CHUNK/sr*1000:.0f} ms rtf={tot/dur:.3f}", flush=True)
        if run > 0:
            firsts.append(first)
            totals.append(tot)
    print(f"{label} MEDIAN first chunk = {statistics.median(firsts)*1000:.0f} ms", flush=True)
    print(f"{label} MEDIAN total       = {statistics.median(totals):.2f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--refs", default=common.DEFAULT_REFS)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--ckpt", default=common.DEFAULT_CKPT)
    ap.add_argument("--vocab", default=common.DEFAULT_VOCAB)
    a = ap.parse_args()

    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    import torchaudio
    r = common.load_refs(a.refs)[a.voice]
    model, voc, acc = common.load_espeech(a.ckpt, a.vocab)
    ra_path, rt = preprocess_ref_audio_text(r["file"], acc.process_all(r["text"]))
    # infer_batch_process expects an already loaded (tensor, sample_rate) tuple
    ra = torchaudio.load(ra_path)

    print("== STREAMING, ONE SEGMENT ==", flush=True)
    stream_runs("RUN", ra, rt, [acc.process_all(TEXT)], model, voc, a.runs)
    print("== STREAMING, TEXT SPLIT INTO PHRASES ==", flush=True)
    stream_runs("RUN2", ra, rt, [acc.process_all(x) for x in PARTS], model, voc, a.runs)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
