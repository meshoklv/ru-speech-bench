# -*- coding: utf-8 -*-
"""Brightness check: how much real energy a recording has above 8 kHz.

Reports the "gap" in dB between the level of the whole recording and the level of
the part above 8 kHz. A SMALLER gap means a wider, brighter sound. It separates a
genuine wideband recording from a 16 kHz-sampled one stretched to a higher rate
(which has nothing above 8 kHz).

Rough reference points measured on the project material:
  12-20 dB  studio-quality wideband recording, bright
  ~23 dB    typical announcer-style sample, fine
  30+ dB    dull (16 kHz source), audibly muffled

Usage:  python3 bandcheck.py file.wav [more files...]
Requires: ffmpeg and ffprobe on PATH.
"""
import os, re, subprocess, sys


def _rms(path, af):
    o = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", af, "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    m = re.search(r"RMS level dB: (-?[\d.]+|-?inf)", o)
    if not m:
        return None
    v = m.group(1)
    return float("-inf") if "inf" in v else float(v)


def sample_rate(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=sample_rate", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def check(path):
    """-> (sample rate, whole-file level dB, level above 8 kHz dB, gap dB)"""
    sr = sample_rate(path)
    full = _rms(path, "astats")
    # resample to 44.1 kHz first: an 8 kHz filter is meaningless on a 16 kHz file otherwise
    hi = _rms(path, "aresample=44100,highpass=f=8000:poles=2,highpass=f=8000:poles=2,astats")
    if full is None or hi is None:
        return sr, full, hi, None
    gap = float("inf") if hi == float("-inf") else round(full - hi, 1)
    return sr, full, hi, gap


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        sr, full, hi, gap = check(p)
        if gap is None:
            print(f"{os.path.basename(p):32s} could not measure")
            continue
        print(f"{os.path.basename(p):32s} {sr:6d} Hz  full {full:7.1f} dB  "
              f"above 8 kHz {hi:7.1f} dB  gap {gap:5.1f} dB")
