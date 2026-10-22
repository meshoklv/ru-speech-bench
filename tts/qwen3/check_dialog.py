"""Round-trip check of synthesized dialogue lines through speech recognition.

Usage:
  STT_CMD="python my_transcribe.py" python check_dialog.py ./out/dialog [--jobs jobs.json]

STT_CMD (env) or --stt-cmd: a command that takes a wav path as its last argument
and prints the transcript on its last stdout line (use any STT engine, e.g. one
from ../../stt). Each line is compared with its source text after normalisation
(lower-case, no punctuation, e->yo folded). Mismatches are reported; extra words at
the start of the transcript are a symptom of reference leakage.
Note: STT writes numbers as digits, so lines with numerals can differ harmlessly.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    s = s.lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9 ]", " ", s).split()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("wav_dir")
    ap.add_argument("--jobs", default=os.path.join(HERE, "jobs.json"))
    ap.add_argument("--stt-cmd", default=os.environ.get("STT_CMD"))
    a = ap.parse_args()
    if not a.stt_cmd:
        sys.exit("set STT_CMD or pass --stt-cmd")
    with open(a.jobs, encoding="utf-8") as f:
        jobs = json.load(f)

    bad = []
    for j in jobs:
        p = os.path.join(a.wav_dir, j["name"] + ".wav")
        if not os.path.exists(p):
            print("MISSING", j["name"])
            bad.append(j["name"])
            continue
        out = subprocess.run(shlex.split(a.stt_cmd) + [p], capture_output=True, text=True,
                             timeout=300).stdout.strip().splitlines()
        got = out[-1] if out else ""
        want, have = norm(j["text"]), norm(got)
        ok = want == have
        print(f"{j['name']}: {'OK  ' if ok else 'DIFF'} | {got}")
        if not ok:
            print(f"    expected: {j['text']}")
            bad.append(j["name"])
    print("BAD:", " ".join(bad) if bad else "none")


if __name__ == "__main__":
    main()
