"""Streaming T-one benchmark (greedy decoder, default ORT threads): 300 ms chunks, phrase-emission lag.

Requires `pip install -e T-one` (voicekit-team/T-one), onnxruntime, huggingface_hub.
Test clips are read from $TESTAUDIO_DIR (default ./testaudio); they are not shipped - supply your own.
Usage: python bench_tone_stream.py [clip_key ...]   (default: a53 b79). Writes results_stream.json.
"""
import os, sys, time, json, resource, statistics as st
from pathlib import Path
import numpy as np
from tone import StreamingCTCPipeline, read_audio
from tone.decoder import DecoderType

AUDIO_DIR = Path(os.environ.get("TESTAUDIO_DIR", "testaudio"))
CHUNK = StreamingCTCPipeline.CHUNK_SIZE
PAD = StreamingCTCPipeline.PADDING
print(f"CHUNK_SIZE={CHUNK} samples = {CHUNK/8000*1000:.0f} ms @8kHz | PADDING={PAD} ({PAD/8000*1000:.0f} ms)")

def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024

pipeline = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)
out = {}
for key in sys.argv[1:] or ["a53", "b79"]:
    for variant in ["16k", "phone8k"]:
        audio = read_audio(AUDIO_DIR / f"{key}-{variant}.wav")
        dur = len(audio)/8000
        audio = np.pad(audio, (PAD, PAD))
        state = None
        chunk_times, lags, first = [], [], None
        t_start = time.perf_counter()
        for i in range(0, len(audio), CHUNK):
            c = audio[i:i+CHUNK]
            c = np.pad(c, (0, -len(c) % CHUNK))
            t = time.perf_counter()
            phrases, state = pipeline.forward(c, state)
            chunk_times.append((time.perf_counter()-t)*1000)
            if phrases:
                audio_pos = (i+CHUNK)/8000 - PAD/8000   # position in the source audio, s
                for p in phrases:
                    lags.append(audio_pos - p.end_time)
                if first is None:
                    first = {"wall_s": round(time.perf_counter()-t_start,2),
                             "audio_pos_s": round(audio_pos,2),
                             "phrase_end_s": round(phrases[0].end_time,2),
                             "lag_s": round(audio_pos - phrases[0].end_time,2),
                             }
        tail, _ = pipeline.finalize(state)
        total = time.perf_counter()-t_start
        ct = sorted(chunk_times)
        r = {"audio_s": round(dur,2), "proc_s": round(total,2), "rtf": round(total/dur,4),
             "speedup": round(dur/total,1), "chunks": len(chunk_times),
             "chunk_ms_mean": round(st.mean(ct),1), "chunk_ms_p50": round(ct[len(ct)//2],1),
             "chunk_ms_p95": round(ct[int(len(ct)*0.95)],1), "chunk_ms_max": round(ct[-1],1),
             "first_phrase": first,
             "lag_mean_s": round(st.mean(lags),2) if lags else None,
             "lag_max_s": round(max(lags),2) if lags else None,
             "tail_phrases": len(tail), "peak_rss_mb": round(rss_mb())}
        out[f"{key}-{variant}"] = r
        print(f"\n[{key}-{variant}] audio {dur:.1f}s, chunks {r['chunks']}")
        print(f"  per 300 ms chunk: mean {r['chunk_ms_mean']} ms | p50 {r['chunk_ms_p50']} | p95 {r['chunk_ms_p95']} | max {r['chunk_ms_max']} ms")
        print(f"  RTF {r['rtf']} (x{r['speedup']} real time), peakRSS {r['peak_rss_mb']} MB")
        print(f"  first phrase: after {first['wall_s']}s of work, audio reached {first['audio_pos_s']}s, phrase ended at {first['phrase_end_s']}s -> LAG {first['lag_s']}s")
        print(f"  phrase emission lag: mean {r['lag_mean_s']}s, max {r['lag_max_s']}s")
Path(__file__).parent.joinpath("results_stream.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("\nsaved results_stream.json")
