#!/usr/bin/env python3
"""Run one STT engine over a list of audio files and record text + timing.

Usage:
    AGGREGATOR_API_KEY=... python run.py <engine> <audio-file> [<audio-file> ...]
    python run.py gigaam clip.ogg            # local engine, no key needed

Engines (names as used in SUMMARY.md): gemini-3.6-flash, gemini-3.5-flash, gemini-3.7-flash,
gemini-3.8-flash, gemini-3.1-pro, gemini-3-pro, gemini-3.6-flash-strict, gigaam.
The Gemini ids are the chat-model ids exposed by the API aggregator (default https://api.kie.ai,
override with AGGREGATOR_BASE_URL); the audio is sent as an mp3 data URI in the chat message.
gigaam calls ../stt_one_shot.py (needs the GigaAM model, see ../../docs/engines/gigaam-v3.md).

Output: ./raw/<file>__<engine>.txt and one JSON line per call appended to ./timing.jsonl
({"eng","file","sec","ok"}). Errors are stored as "ERR: ..." in the raw file. Keys are read from
the environment only and never printed.
"""
import base64, json, os, subprocess, sys, tempfile, time, urllib.request, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("AGGREGATOR_BASE_URL", "https://api.kie.ai").rstrip("/")
TIMEOUT = 180

# The plain prompt below is a paraphrase of the one used in the original run (the original
# string was a verbatim-transcription instruction in Russian).
PLAIN = ("This is a voice message in Russian. Transcribe it verbatim. "
         "Return only the transcript, no comments.")
STRICT = ("Это голосовое сообщение на русском языке. Расшифруй его дословно, слово в слово, "
          "ничего не добавляй, не додумывай и не пересказывай. Если слово не слышно — пиши [неразборчиво]. "
          "Английские названия пиши как сказано. Верни только текст.")


def to_mp3(path):
    out = os.path.join(tempfile.gettempdir(), "cmp-%s.mp3" % uuid.uuid4().hex)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-vn", "-i", path,
                    "-ac", "1", "-ar", "16000", "-b:a", "64k", "-y", out], check=True)
    return out


def chat_model(path, model, prompt=PLAIN):
    key = os.environ["AGGREGATOR_API_KEY"]
    mp3 = to_mp3(path)
    with open(mp3, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.unlink(mp3)
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        # the aggregator accepts audio as a data URI in an image_url part
        {"type": "image_url", "image_url": {"url": "data:audio/mpeg;base64," + b64}}]}]}).encode()
    req = urllib.request.Request(
        "%s/%s/v1/chat/completions" % (BASE, model), data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "User-Agent": "curl/8.5.0"})
    d = json.load(urllib.request.urlopen(req, timeout=TIMEOUT))
    if "choices" not in d:
        raise RuntimeError(str(d)[:150])
    return d["choices"][0]["message"]["content"].strip()


def gigaam(path):
    p = subprocess.run([sys.executable, os.path.join(HERE, "..", "stt_one_shot.py"), path],
                       capture_output=True, timeout=360)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-200:])
    return p.stdout.decode("utf-8", "replace").strip()


ENGINES = {
    "gemini-3.6-flash": lambda p: chat_model(p, "gemini-3-6-flash-openai"),
    "gemini-3.5-flash": lambda p: chat_model(p, "gemini-3-5-flash-openai"),
    "gemini-3.7-flash": lambda p: chat_model(p, "gemini-3-7-flash-openai"),
    "gemini-3.8-flash": lambda p: chat_model(p, "gemini-3-8-flash-openai"),
    "gemini-3.1-pro": lambda p: chat_model(p, "gemini-3.1-pro"),
    "gemini-3-pro": lambda p: chat_model(p, "gemini-3-pro"),
    "gemini-3.6-flash-strict": lambda p: chat_model(p, "gemini-3-6-flash-openai", STRICT),
    "gigaam": gigaam,
}

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ENGINES:
        sys.exit("usage: run.py <%s> <audio-file>..." % "|".join(ENGINES))
    eng = sys.argv[1]
    os.makedirs(os.path.join(HERE, "raw"), exist_ok=True)
    for f in sys.argv[2:]:
        base = os.path.splitext(os.path.basename(f))[0]
        t = time.time()
        try:
            text, ok = ENGINES[eng](f), True
        except Exception as e:
            text, ok = "ERR: %s" % str(e)[:200], False
        dt = time.time() - t
        with open(os.path.join(HERE, "raw", "%s__%s.txt" % (base, eng)), "w", encoding="utf-8") as o:
            o.write(text)
        with open(os.path.join(HERE, "timing.jsonl"), "a") as o:
            o.write(json.dumps({"eng": eng, "file": base, "sec": round(dt, 1), "ok": ok}) + "\n")
        print(eng, base, round(dt, 1), "OK" if ok else text, flush=True)
