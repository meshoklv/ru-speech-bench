#!/usr/bin/env python3
"""Speech generation through kie.ai (jobs/createTask -> poll recordInfo -> download).

The API key is read ONLY from the KIE_API_KEY environment variable (never argv).
Measures: time until the task is ready, creditsConsumed (as reported by the
service), and duration of the resulting audio.

  export KIE_API_KEY=...
  python3 kie_tts.py --model google/gemini-2-5-pro-tts --input @input.json --out a.wav

--input is a JSON string or @file holding the model's "input" object (see
docs/engines/kie-ai-tts.md for an example). ffprobe is optional (duration probe).
"""
import json, os, ssl, sys, time, urllib.request, urllib.error, argparse, subprocess

API = "https://api.kie.ai/api/v1/jobs"
KEY = os.environ.get("KIE_API_KEY")


def _req(url, data=None, method=None):
    hdr = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, headers=hdr, method=method)
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read())


def credits():
    return _req("https://api.kie.ai/api/v1/chat/credit")["data"]


def generate(model, inp, out_path, poll=1.5, timeout=300):
    t0 = time.time()
    r = _req(f"{API}/createTask", {"model": model, "input": inp})
    if r.get("code") != 200:
        raise RuntimeError(f"createTask: {r}")
    task_id = r["data"]["taskId"]
    info = None
    while time.time() - t0 < timeout:
        time.sleep(poll)
        info = _req(f"{API}/recordInfo?taskId={task_id}")
        d = info.get("data") or {}
        state = d.get("state") or d.get("status")
        if state in ("success", "fail", "failed", "error"):
            break
    t_gen = time.time() - t0
    d = info.get("data") or {}
    if (d.get("state") or d.get("status")) != "success":
        raise RuntimeError(f"task failed: {json.dumps(info, ensure_ascii=False)[:800]}")
    res = d.get("resultJson")
    res = json.loads(res) if isinstance(res, str) else (res or {})
    urls = res.get("resultUrls") or res.get("result_urls") or []
    if not urls:
        raise RuntimeError(f"no audio url: {json.dumps(res, ensure_ascii=False)[:500]}")
    t1 = time.time()
    dl = urllib.request.Request(urls[0], headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
        "Referer": "https://kie.ai/",
    })
    try:
        with urllib.request.urlopen(dl, timeout=120) as a:
            audio = a.read()
    except urllib.error.HTTPError as e:
        # download refused -- the task is already billed; surface the URL for manual retrieval
        raise RuntimeError(f"download {e.code} for {urls[0]}") from None
    open(out_path, "wb").write(audio)
    return {
        "task_id": task_id,
        "url": urls[0],
        "gen_sec": round(t_gen, 2),
        "dl_sec": round(time.time() - t1, 2),
        "credits": d.get("creditsConsumed"),
        "cost_time_ms": d.get("costTime"),
        "bytes": len(audio),
        "file": out_path,
    }


def dur(path):
    try:
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
        return round(float(o.stdout.strip()), 2)
    except Exception:
        return None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, help="JSON string or @file")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    inp = a.input
    if inp.startswith("@"):
        inp = open(inp[1:]).read()
    r = generate(a.model, json.loads(inp), a.out)
    r["audio_sec"] = dur(a.out)
    print(json.dumps(r, ensure_ascii=False))
