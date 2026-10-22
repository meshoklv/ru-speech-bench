#!/usr/bin/env python3
"""Measure Fish Audio streaming: time to first audio chunk, chunk arrival times, total time.

The API key is read from the FISH_API_KEY environment variable (never from argv).
Prints one JSON object, including the Server-Timing split (edge vs. origin) when
the server returns it. The request deliberately sends no "Accept: */*" header,
because with it Cloudflare omits Server-Timing.

  export FISH_API_KEY=...
  python3 fish_stream_bench.py --text "Hello" [--voice <reference_id>] \
          --model s2.1-pro-free --latency low [--out out.mp3]

A new TLS connection is opened per run, so absolute numbers include ~0.6 s of
connection setup that a long-lived client would not pay.
"""
import argparse, http.client, json, os, sys, time

def key():
    k = os.environ.get("FISH_API_KEY")
    if not k:
        raise SystemExit("set the FISH_API_KEY environment variable")
    return k


def run(text, voice, model, latency, fmt, out, sample_rate=None):
    body = {"text": text, "format": fmt, "latency": latency, "normalize": True}
    if voice:
        body["reference_id"] = voice
    if sample_rate:
        body["sample_rate"] = sample_rate
    payload = json.dumps(body).encode()
    c = http.client.HTTPSConnection("api.fish.audio", timeout=300)
    t0 = time.time()
    c.request("POST", "/v1/tts", body=payload, headers={
        "Authorization": "Bearer " + key(), "Content-Type": "application/json",
        "model": model})   # no "Accept: */*": with it Cloudflare does not return Server-Timing
    r = c.getresponse()
    t_hdr = time.time() - t0
    if r.status != 200:
        print(json.dumps({"http": r.status, "body": r.read()[:300].decode("utf-8", "replace")},
                         ensure_ascii=False)); return
    st = " ".join(v for kk, v in r.getheaders() if kk.lower() == "server-timing")
    import re as _re
    _e = _re.search(r"cfEdge;dur=(\d+)", st); _o = _re.search(r"cfOrigin;dur=(\d+)", st)
    edge = int(_e.group(1))/1000 if _e else None      # time to the network edge
    origin = int(_o.group(1))/1000 if _o else None    # time spent by the Fish origin
    chunks, sizes, blob = [], [], b""
    while True:
        b = r.read1(65536)
        if not b:
            break
        chunks.append(round(time.time() - t0, 3)); sizes.append(len(b)); blob += b
    total = time.time() - t0
    c.close()
    if out:
        open(out, "wb").write(blob)
    print(json.dumps({
        "model": model, "latency": latency, "format": fmt,
        "ttfb_header_s": round(t_hdr, 3),
        "edge_s": edge, "fish_origin_s": origin,
        "ttfa_first_chunk_s": chunks[0] if chunks else None,
        "total_s": round(total, 3), "n_chunks": len(chunks),
        "chunk_times_s": chunks[:12], "chunk_sizes": sizes[:12],
        "bytes": len(blob), "utf8_in": len(text.encode()), "out": out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--voice")
    p.add_argument("--model", default="s2.1-pro-free")
    p.add_argument("--latency", default="normal")
    p.add_argument("--format", default="mp3")
    p.add_argument("--sample-rate", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    run(a.text, a.voice, a.model, a.latency, a.format, a.out, a.sample_rate)
