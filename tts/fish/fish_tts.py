#!/usr/bin/env python3
"""Text-to-speech through Fish Audio (api.fish.audio).

The API key is read from the FISH_API_KEY environment variable only (never argv).

  export FISH_API_KEY=...

  # find Russian voices in the public library
  python3 fish_tts.py --search --language ru --limit 20

  # synthesise a file of lines (one line per utterance) with a chosen voice
  python3 fish_tts.py --voice <reference_id> --text-file phrases.txt --out-dir ./out

  # single phrase
  python3 fish_tts.py --voice <reference_id> --text "Hello" --out a.mp3

Cost note: billing is per UTF-8 BYTE. A Cyrillic letter is 2 bytes, so Russian
text costs about twice as much per character as English. The script prints bytes
in and a cost estimate at the rate set in FISH_USD_PER_MB (USD per 1M bytes,
default 15 -- the rate seen on the vendor pricing page on 2026-09-15; check the
current page before relying on it).
"""
import argparse, json, os, time, urllib.parse, urllib.request

BASE = "https://api.fish.audio"


def key():
    k = os.environ.get("FISH_API_KEY")
    if not k:
        raise SystemExit("set the FISH_API_KEY environment variable")
    return k


def search(language=None, title=None, limit=20):
    q = {"page_size": limit, "page_number": 1, "sort_by": "task_count"}
    if language:
        q["language"] = language
    if title:
        q["title"] = title
    req = urllib.request.Request(f"{BASE}/model?{urllib.parse.urlencode(q)}",
                                 headers={"Authorization": "Bearer " + key()})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    for m in d.get("items", d.get("data", [])):
        print(f"{m.get('_id') or m.get('id')}  {m.get('title','')!r:40} "
              f"languages={m.get('languages')} tasks={m.get('task_count')}")


def tts(text, voice, out, model="s2.1-pro-free", fmt="mp3", latency="normal",
        sample_rate=None, temperature=0.7, top_p=0.7):
    body = {"text": text, "format": fmt, "latency": latency,
            "temperature": temperature, "top_p": top_p, "normalize": True}
    if voice:
        body["reference_id"] = voice
    if sample_rate:
        body["sample_rate"] = sample_rate
    req = urllib.request.Request(f"{BASE}/v1/tts", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key(),
                                          "Content-Type": "application/json",
                                          "model": model})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        blob = r.read()
    open(out, "wb").write(blob)
    nbytes = len(text.encode())
    return {"file": out, "gen_sec": round(time.time() - t0, 2), "bytes_out": len(blob),
            "utf8_bytes_in": nbytes, "usd": round(nbytes * float(os.environ.get("FISH_USD_PER_MB", "15")) / 1e6, 5),
            "model": model, "latency": latency}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--search", action="store_true")
    p.add_argument("--language")
    p.add_argument("--title")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--voice", help="reference_id of a voice from the library")
    p.add_argument("--text")
    p.add_argument("--text-file", help="one utterance per line; writes k1.mp3, k2.mp3, ...")
    p.add_argument("--out", default="out.mp3")
    p.add_argument("--out-dir")
    p.add_argument("--model", default="s2.1-pro-free",
                   help="s2.1-pro-free (free tier) | s2.1-pro | s2-pro | s1")
    p.add_argument("--format", default="mp3", choices=["mp3", "wav", "pcm", "opus"])
    p.add_argument("--latency", default="normal", choices=["normal", "balanced", "low"])
    a = p.parse_args()
    if a.search:
        search(a.language, a.title, a.limit)
    elif a.text_file:
        os.makedirs(a.out_dir or ".", exist_ok=True)
        for i, line in enumerate((l.strip() for l in open(a.text_file, encoding="utf-8")), 1):
            if not line:
                continue
            o = os.path.join(a.out_dir or ".", f"k{i}.{a.format}")
            print(json.dumps(tts(line, a.voice, o, a.model, a.format, a.latency),
                             ensure_ascii=False), flush=True)
    elif a.text:
        print(json.dumps(tts(a.text, a.voice, a.out, a.model, a.format, a.latency),
                         ensure_ascii=False))
    else:
        p.error("one of --search, --text or --text-file is required")
