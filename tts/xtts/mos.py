"""Objective naturalness estimate with UTMOS (a MOS predictor).

Usage:
  python mos.py ./out/xtts ./out/f5 [--json ./out/mos.json]

The model listens to each wav and outputs a 1-5 score, as listeners in a blind test would.
CAUTION: it was trained on English data (VCC), so for Russian it is an APPROXIMATE
indicator - fine for comparing engines with each other, not an absolute "quality of Russian
speech". It also measures cleanliness, not intonation, accent or pauses.
Files are grouped by the file-name prefix before the last '-' (engine+voice) for averages.
Needs: GPU, `pip install torch torchaudio` (SpeechMOS is fetched via torch.hub: tarepan/SpeechMOS v1.2.0).
"""
import argparse
import glob
import json
import os
import statistics

os.environ.setdefault("TORCH_HOME", "./models/torchhub")

import torch
import torchaudio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="directories with wavs")
    ap.add_argument("--json", default="./out/mos.json")
    a = ap.parse_args()

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.cuda().eval()
    out = {}
    for d in a.dirs:
        scores = {}
        for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
            wav, sr = torchaudio.load(p)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = torchaudio.functional.resample(wav, sr, 16000)
            with torch.no_grad():
                s = float(predictor(wav.cuda(), 16000))
            scores[os.path.basename(p)] = round(s, 3)
            print(f"{os.path.basename(p):44s} MOS {s:.2f}", flush=True)
        out.update(scores)
        groups = {}
        for k, v in scores.items():
            groups.setdefault("-".join(k.split("-")[:-1]), []).append(v)
        for g, vs in sorted(groups.items()):
            print(f"  [mean] {g:40s} {statistics.mean(vs):.2f}  (n={len(vs)})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[ok] {a.json}", flush=True)


if __name__ == "__main__":
    main()
