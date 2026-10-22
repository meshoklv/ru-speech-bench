#!/usr/bin/env python3
"""Synthesis with streaming AND word timestamps (SSE /v1/tts/stream/with-timestamp).

Prints JSON: latency to the first event, word boundaries, output path.
The API key is read from the FISH_API_KEY environment variable.

  export FISH_API_KEY=...
  python3 fish_timestamps.py --text "Hello" [--voice <reference_id>] --out out.mp3
"""
import argparse, base64, http.client, json, os, time

def key():
    k = os.environ.get("FISH_API_KEY")
    if not k:
        raise SystemExit("set the FISH_API_KEY environment variable")
    return k

def run(text, voice, model, latency, fmt, out):
    body = {"text": text, "format": fmt, "latency": latency}
    if voice: body["reference_id"] = voice
    c = http.client.HTTPSConnection("api.fish.audio", timeout=300)
    t0 = time.time()
    c.request("POST", "/v1/tts/stream/with-timestamp", body=json.dumps(body).encode(),
              headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json",
                       "model": model, "Accept": "text/event-stream"})
    r = c.getresponse()
    if r.status != 200:
        print(json.dumps({"http": r.status, "body": r.read()[:300].decode("utf-8","replace")}, ensure_ascii=False)); return
    audio, segs, first, buf = b"", {}, None, b""
    while True:
        b = r.read1(65536)
        if not b: break
        buf += b
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            block = ev.decode("utf-8", "replace")
            payload = "".join(l[5:].strip() for l in block.splitlines() if l.startswith("data:"))
            if payload in ("[DONE]", ""): continue
            try: d = json.loads(payload)
            except Exception: continue
            if first is None: first = round(time.time() - t0, 3)
            if d.get("audio_base64"): audio += base64.b64decode(d["audio_base64"])
            if d.get("alignment"):
                segs[d.get("chunk_seq", 0)] = (d.get("chunk_audio_offset_sec", 0.0),
                                               d["alignment"].get("segments", []))
    c.close()
    if out: open(out, "wb").write(audio)
    words = []
    for seq in sorted(segs):
        off, ss = segs[seq]
        for s in ss:
            words.append({"t": s.get("text"), "start": round(off + s.get("start", 0), 3),
                          "end": round(off + s.get("end", 0), 3)})
    print(json.dumps({"ttfa_first_event_s": first, "total_s": round(time.time()-t0, 3),
                      "bytes": len(audio), "out": out, "words": words}, ensure_ascii=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True); p.add_argument("--voice")
    p.add_argument("--model", default="s2.1-pro-free"); p.add_argument("--latency", default="low")
    p.add_argument("--format", default="mp3"); p.add_argument("--out")
    a = p.parse_args(); run(a.text, a.voice, a.model, a.latency, a.format, a.out)
