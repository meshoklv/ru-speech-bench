# -*- coding: utf-8 -*-
"""Pick lively conversational reference clips from SOVA RuDevices (CC-BY-4.0) for voice cloning.

Key lesson: select references by INTONATION LIVELINESS (pitch spread), not by
audio cleanliness. A cloning engine reproduces roughly 80% of a reference's pitch
spread, so a clean but monotone clip yields a dead-sounding voice. Prefer clips
with a pitch spread of at least ~4.5 semitones.

Clips are fetched one by one through the HuggingFace datasets-server API; the full
dataset is never downloaded. Output goes to ./live_pick (or the 3rd argument).

Usage:  python3 pick_live_refs.py [rows_to_scan=1200] [clips_to_take=6] [workdir=live_pick]
Requires: ffmpeg/ffprobe on PATH, network access to huggingface.co, pitchstat.py
next to this file. Attribute the dataset (CC-BY-4.0) if you redistribute derived audio.
"""
import json, os, re, subprocess, sys, time, urllib.parse, urllib.request

DATASET = "bond005/sova_rudevices"
# profanity filter: Russian-language data, intentionally left in Cyrillic
BAD = re.compile(r'ху[йея]|пизд|бля|еба|сук[аи]|говн|срат|рыга|нахер|нахуй|пидор|дерьм|жоп|пипец', re.I)
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
from pitchstat import stats                      # pitch-spread measurement

def api(url):
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(url, timeout=90))
        except Exception as e: err = str(e)[:150]; time.sleep(3)
    return {"err": err}

def fetch_rows(split, limit):
    rows = []
    for off in range(0, limit, 100):
        r = api(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(DATASET)}"
                f"&config=default&split={split}&offset={off}&length=100")
        if "err" in r: print("API:", r["err"]); break
        for row in r["rows"]:
            rr = row["row"]; a = rr["audio"]
            rows.append(dict(idx=row["row_idx"], text=rr["transcription"],
                             src=(a[0]["src"] if isinstance(a, list) and a else None),
                             nw=len(rr["transcription"].split())))
    return rows

def snr_of(path):
    """Speech level minus pause level: a rough cleanliness (SNR) estimate."""
    o = subprocess.run(["ffmpeg","-hide_banner","-nostats","-i",path,"-af","astats","-f","null","-"],
                       capture_output=True, text=True).stderr
    g = lambda k: float(re.search(rf"{k}: (-?[\d.]+)", o).group(1)) if re.search(rf"{k}: (-?[\d.]+)", o) else 0.0
    return round(g("RMS level dB") - g("RMS trough dB"), 1)

def duration(path):
    return float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",path],
                                capture_output=True, text=True).stdout.strip() or 0)

def main(scan=1200, take=6, split="validation", workdir="live_pick"):
    os.makedirs(workdir, exist_ok=True)
    rows = fetch_rows(split, scan)
    print(f"rows scanned: {len(rows)}")
    cand = [r for r in rows if r["src"] and not BAD.search(r["text"]) and r["nw"] >= 14]
    cand.sort(key=lambda r: -r["nw"])
    scored = []
    for r in cand[:60]:
        f = os.path.join(workdir, f"sova-{r['idx']:04d}.wav")
        if not os.path.exists(f):
            try: urllib.request.urlretrieve(r["src"], f)
            except Exception: continue
        d = duration(f)
        if not 5 <= d <= 12:        # the ESpeech model card asks for references of at most 12 s
            continue
        s = stats(f)
        if not s: continue
        snr = snr_of(f)
        if snr < 25 or s["voiced"] < 0.35:
            continue
        scored.append(dict(idx=r["idx"], file=f, text=r["text"], dur=round(d,1), snr=snr, **s))
        print(".", end="", flush=True)
    print()
    # primary sort key: pitch liveliness; cleanliness is secondary
    scored.sort(key=lambda r: -(r["sd_st"]*0.5 + r["rng_st"]*0.15))
    print(f"\n{'idx':>5s} {'dur':>4s} {'snr':>5s} {'Hz':>7s} {'sd(st)':>8s} {'rng':>6s} | text")
    for r in scored[:take*3]:
        pol = "F" if r["med"] >= 165 else "M"
        print(f"{r['idx']:5d} {r['dur']:4.1f} {r['snr']:5.1f} {r['med']:6.1f}{pol} "
              f"{r['sd_st']:8.2f} {r['rng_st']:6.2f} | {r['text'][:62]}")
    json.dump(scored, open(os.path.join(workdir, "candidates.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\nDetails: {workdir}/candidates.json")
    print("Prefer clips with a pitch spread of 4.5 semitones or more. Check that the transcript matches")
    print("the audio: the engine takes both audio and reference text, and a wrong transcript degrades the clone.")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1200,
         int(sys.argv[2]) if len(sys.argv) > 2 else 6,
         workdir=sys.argv[3] if len(sys.argv) > 3 else "live_pick")
