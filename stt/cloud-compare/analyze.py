#!/usr/bin/env python3
"""Consensus-distance and timing summary for the cloud-vs-local STT comparison.

Usage: python analyze.py [raw_dir] [timing.jsonl]      (defaults: ./raw, ./timing.jsonl)

For every clip, each engine's transcript is compared with every other engine's transcript
(word-level edit distance divided by the other transcript's length; filler words such as
"э-э", "м-м" and punctuation are dropped). The per-engine score is the mean over clips and
over the other engines. This is NOT a WER: there is no human reference. It only shows which
engine deviates from the consensus of the others. Median time is taken over successful calls.
"""
import collections, glob, json, os, re, sys

RAW = sys.argv[1] if len(sys.argv) > 1 else "raw"
TIMING = sys.argv[2] if len(sys.argv) > 2 else "timing.jsonl"


def norm(t):
    t = t.lower().replace("ё", "е")
    t = re.sub(r"[^\w\s]", " ", t)
    return [w for w in t.split() if not re.fullmatch(r"(э+м*|э+|а+|м+|эм+|ээ+|о+)", w)]


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


files = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(RAW, "*__*"))})
eng = collections.defaultdict(dict)
for f in glob.glob(os.path.join(RAW, "*__*")):
    clip, e = os.path.basename(f)[:-4].split("__")
    text = open(f, encoding="utf-8").read().strip()
    if text and not text.startswith("ERR"):
        eng[e][clip] = text

dist = collections.defaultdict(list)
for clip in files:
    toks = {e: norm(eng[e][clip]) for e in eng if clip in eng[e]}
    for e in toks:
        ds = [edit(toks[e], toks[o]) / max(len(toks[o]), 1) for o in toks if o != e]
        if ds:
            dist[e].append((clip, sum(ds) / len(ds)))

times = collections.defaultdict(list)
calls = collections.Counter()
for line in open(TIMING):
    d = json.loads(line)
    calls[d["eng"]] += 1
    if d["ok"]:
        times[d["eng"]].append(d["sec"])

print("engine  ok/total  mean_dist_to_others  median_sec")
for e, v in sorted(dist.items(), key=lambda kv: sum(x for _, x in kv[1]) / len(kv[1])):
    ts = sorted(times[e])
    med = ts[len(ts) // 2] if ts else "-"
    print(f"{e:24s} {len(v)}/{len(files)}  {100 * sum(x for _, x in v) / len(v):5.1f}%  {med}s  "
          + " ".join(f"{c}:{100 * x:.0f}" for c, x in v))
