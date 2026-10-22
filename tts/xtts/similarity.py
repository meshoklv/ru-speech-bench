"""How similar is a synthesized wav to the reference voice - objectively, not by ear.

Usage:
  COQUI_TOS_AGREED=1 python similarity.py --refs-dir ./refs ./out/xtts ./out/f5 [--json ./out/similarity.json]

Uses the speaker encoder of XTTS itself (H/ASP, downloaded with the model): embeds every
reference (*.wav in --refs-dir) and every wav in the given directories and prints the
cosine similarity. 1.0 = same voice, ~0 = nothing in common. Such encoders usually treat
0.6-0.7 as "recognisably the same person", but absolute values depend on the encoder, so
compare engines by the DIFFERENCE. The encoder is XTTS's own, so it can favour XTTS.
Needs: GPU, `pip install coqui-tts "transformers<5"`.
"""
import argparse
import glob
import json
import os

os.environ.setdefault("TTS_HOME", "./models")

import torch
import torchaudio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="directories with synthesized wavs")
    ap.add_argument("--refs-dir", default="./refs")
    ap.add_argument("--json", default="./out/similarity.json")
    a = ap.parse_args()

    from TTS.utils.manage import ModelManager
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    mm = ModelManager(output_prefix=os.environ["TTS_HOME"], progress_bar=False)
    model_path, _, _ = mm.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    cfg = XttsConfig()
    cfg.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=model_path, eval=True, use_deepspeed=False)
    model.cuda()

    def emb(path):
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = torchaudio.functional.resample(wav, sr, 16000)
        return model.hifigan_decoder.speaker_encoder.forward(wav.cuda(), l2_norm=True).squeeze()

    refs = {os.path.basename(p): emb(p) for p in glob.glob(os.path.join(a.refs_dir, "*.wav"))}
    print("[references]", ", ".join(refs), flush=True)

    out = {}
    for d in a.dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
            e = emb(p)
            row = {rn: round(float(torch.nn.functional.cosine_similarity(e, re_, dim=0)), 3)
                   for rn, re_ in refs.items()}
            out[os.path.basename(p)] = row
            best = max(row, key=row.get)
            print(f"{os.path.basename(p):44s} " +
                  "  ".join(f"{k.replace('.wav',''):12s}={v:+.3f}" for k, v in row.items()) +
                  f"   -> closest: {best}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[ok] {a.json}", flush=True)


if __name__ == "__main__":
    main()
