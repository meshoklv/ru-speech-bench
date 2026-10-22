"""Round-trip check of a synthesized dialogue through speech recognition, with a
reference-leakage heuristic (the transcript starts with words that are not in the text).

Usage:
  STT_CMD="python my_transcribe.py" python check_zv.py ./out/dialog [--jobs qwen3/jobs.json]

STT_CMD (env) or --stt-cmd: command that takes a wav path as its last argument and prints
the transcript on its last stdout line. Output per line: OK / LEAK? / ok(numerals).
"numerals" = text differs but the first word matches (STT writes numbers as digits).
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
    ap.add_argument("--jobs", default=os.path.join(HERE, "qwen3", "jobs.json"))
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
        try:
            out = subprocess.run(shlex.split(a.stt_cmd) + [p], capture_output=True, text=True,
                                 timeout=300).stdout.strip().splitlines()
            got = out[-1] if out else ""
        except Exception as e:
            got = "ERR " + str(e)
        want, have = norm(j["text"]), norm(got)
        ok = want == have
        leak = have[0] not in want[:2] if have else True  # extra words at the START = leakage
        flag = "OK    " if ok else ("LEAK?" if leak else "ok(numerals)")
        print(f"{j['name']}: {flag} | {got}")
        if not ok and leak:
            print(f"    expected: {j['text']}")
            bad.append(j["name"])
    print("PROBLEMATIC:", " ".join(bad) if bad else "none")


if __name__ == "__main__":
    main()
