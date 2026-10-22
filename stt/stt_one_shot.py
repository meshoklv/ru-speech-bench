"""One call = one GigaAM transcription of audio in ANY format.

Usage:
    python stt_one_shot.py <audio-file> [--dialog]

Converts the input to mono 16 kHz wav with ffmpeg and runs gigaam_serve.py in a subprocess.
Text goes to stdout; errors go to stderr with a non-zero exit code.
--dialog: for a stereo call recording (left channel = agent, right channel = customer) the
channels are transcribed separately, merged by timestamp and printed as JSON
{"phrases": [["operator"|"client", text], ...]}. Mono input falls back to plain text.

Requires: ffmpeg/ffprobe on PATH, and the environment of gigaam_serve.py. The interpreter used
for the subprocess is $GIGAAM_PYTHON (default: the current interpreter).
"""
import difflib
import json
import os
import subprocess
import sys
import tempfile
import uuid

STT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("GIGAAM_PYTHON", sys.executable)

# Calls can be 20-30 minutes; at ~6x real time that is up to ~5 min of compute plus headroom.
# A smaller limit would cut a long call off mid-sentence.
DECODE_TIMEOUT_S = 360


def _run_gigaam(wav, out_json=None):
    cmd = [PYTHON, os.path.join(STT_DIR, "gigaam_serve.py"), wav, "--threads", "4"]
    if out_json:
        cmd += ["--json", out_json]
    p = subprocess.run(cmd, capture_output=True, timeout=DECODE_TIMEOUT_S, cwd=STT_DIR)
    if p.returncode != 0:
        sys.exit((p.stderr or b"").decode("utf-8", "replace").strip()[-200:] or "GigaAM failed")
    return (p.stdout or b"").decode("utf-8", "replace").strip()


def _channels(src):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=channels", "-of", "csv=p=0", src], capture_output=True, timeout=60)
    try:
        return int((p.stdout or b"0").decode().strip() or 0)
    except ValueError:
        return 0


def _echo(a, b):
    """The same audio leaking into the other channel (answering machine, echo)."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= 0.85


def dialog(src):
    """Stereo call: LEFT channel = agent, RIGHT channel = customer. Channels are decoded
    separately and merged by time into turns. Prints JSON {"phrases": [[role, text], ...]}."""
    tag = uuid.uuid4().hex
    tmp = tempfile.gettempdir()
    left, right = (os.path.join(tmp, f"stt1-{tag}-{c}.wav") for c in "LR")
    try:
        c = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                            "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[L][R]",
                            "-map", "[L]", "-ac", "1", "-ar", "16000", left,
                            "-map", "[R]", "-ac", "1", "-ar", "16000", right],
                           capture_output=True, timeout=300)
        if c.returncode != 0:
            sys.exit("ffmpeg could not split channels: " + (c.stderr or b"").decode("utf-8", "replace")[-200:])
        items = []
        for who, wav in (("operator", left), ("client", right)):
            _run_gigaam(wav, wav + ".json")
            with open(wav + ".json", encoding="utf-8") as f:
                items += [(who, seg) for seg in json.load(f)]
        items.sort(key=lambda it: (it[1]["start"], it[1]["end"]))
        phrases = []
        for who, seg in items:
            text = seg["text"].strip()
            if not text:
                continue
            if phrases and phrases[-1][0] == who:
                phrases[-1][1] += " " + text
            elif phrases and _echo(phrases[-1][1], text):
                continue
            else:
                phrases.append([who, text])
        if not phrases:
            sys.exit("empty result")
        print(json.dumps({"phrases": phrases}, ensure_ascii=False))
    finally:
        for f in (left, right, left + ".json", right + ".json"):
            try:
                os.unlink(f)
            except OSError:
                pass


def main():
    args = [a for a in sys.argv[1:] if a != "--dialog"]
    if not args or not os.path.isfile(args[0]):
        sys.exit("usage: stt_one_shot.py <audio-file> [--dialog]")
    src = args[0]
    if "--dialog" in sys.argv and _channels(src) >= 2:
        dialog(src)
        return
    wav = os.path.join(tempfile.gettempdir(), "stt1-%s.wav" % uuid.uuid4().hex)
    try:
        c = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-vn",
                            "-i", src, "-ac", "1", "-ar", "16000", "-y", wav],
                           capture_output=True, timeout=300)
        if c.returncode != 0 or not os.path.getsize(wav):
            sys.exit("ffmpeg failed: " + (c.stderr or b"").decode("utf-8", "replace")[-200:])
        print(_run_gigaam(wav) or sys.exit("empty result"))
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass


if __name__ == "__main__":
    main()
