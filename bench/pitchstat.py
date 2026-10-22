# -*- coding: utf-8 -*-
"""Pitch spread in semitones as an objective measure of intonation "liveliness".

A monotone, announcer-style delivery has a narrow, flat pitch distribution; lively
speech is wide and jagged. Pitch is estimated by autocorrelation (pure stdlib +
ffmpeg for resampling to 8 kHz mono), unvoiced/silent frames are dropped by energy.

Reported per file:
  sd_st   standard deviation of voiced pitch, in semitones
  rng_st  10th-90th percentile range of voiced pitch, in semitones
  med     median F0 in Hz
  voiced  fraction of frames that are voiced

Usage:
  python3 pitchstat.py a.wav b.wav ...                  # per-file stats + mean
  python3 pitchstat.py --group "label=glob/*.wav" --group "other=dir/*.wav" \
                       [--json out.json]                # per-group means

Requires: ffmpeg on PATH. Not a substitute for listening: it does not detect
hoarseness, breathiness or synthesis artefacts.
"""
import argparse, array, glob, json, math, os, statistics, subprocess, tempfile, wave


def to8k(path):
    out = os.path.join(tempfile.gettempdir(),
                       "p8_" + os.path.splitext(os.path.basename(path))[0] + ".wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "8000", out],
                   check=True)
    return out


def f0track(path):
    w = wave.open(to8k(path))
    sr = w.getframerate()
    a = array.array('h')
    a.frombytes(w.readframes(w.getnframes()))
    w.close()
    N, HOP = 1024, 256          # 128 ms frame, 32 ms hop
    lo, hi = int(sr / 400), int(sr / 70)
    track, energies = [], []
    for s in range(0, len(a) - N, HOP):
        seg = [float(x) for x in a[s:s + N]]
        m = sum(seg) / N
        seg = [x - m for x in seg]
        r0 = sum(x * x for x in seg)
        energies.append(r0 / N)
        if r0 <= 1e-6:
            track.append(None)
            continue
        best = (0.0, 0)
        for lag in range(lo, hi):
            acc = 0.0
            for i in range(0, N - lag, 2):      # stride 2: twice as fast, no accuracy loss
                acc += seg[i] * seg[i + lag]
            r = acc * 2 / ((N - lag) * (r0 / N))
            if r > best[0]:
                best = (r, lag)
        track.append(sr / best[1] if best[0] > 0.35 and best[1] else None)
    if energies:                                  # voiced-frame gate by energy
        thr = max(energies) * 0.02
        track = [f if (f and energies[i] > thr) else None for i, f in enumerate(track)]
    return track


def stats(path):
    t = f0track(path)
    v = [f for f in t if f]
    if len(v) < 8:
        return None
    st = sorted(12 * math.log2(f) for f in v)     # semitones
    p10, p90 = st[int(len(st) * 0.10)], st[int(len(st) * 0.90)]
    return dict(n=len(v), voiced=round(len(v) / len(t), 2),
                med=round(statistics.median(v), 1),
                sd_st=round(statistics.pstdev(st), 2),
                rng_st=round(p90 - p10, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--group", action="append", default=[], metavar="LABEL=GLOB")
    ap.add_argument("--json", help="write per-file results to this path")
    a = ap.parse_args()
    groups = {}
    if a.files:
        groups["files"] = a.files
    for g in a.group:
        label, _, pattern = g.partition("=")
        groups[label] = sorted(glob.glob(pattern))
    if not groups:
        ap.error("give files or --group LABEL=GLOB")
    allres = {}
    for g, files in groups.items():
        rows = []
        for f in files:
            s = stats(f)
            if s:
                s["file"] = os.path.basename(f)
                rows.append(s)
        allres[g] = rows
    print(f"{'group':40s} {'n':>3s} {'median Hz':>10s} {'sd (st)':>8s} {'range (st)':>11s}")
    for g, rows in allres.items():
        if not rows:
            continue
        print(f"{g:40s} {len(rows):3d} {statistics.mean(r['med'] for r in rows):10.1f} "
              f"{statistics.mean(r['sd_st'] for r in rows):8.2f} "
              f"{statistics.mean(r['rng_st'] for r in rows):11.2f}")
    if a.json:
        json.dump(allres, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
