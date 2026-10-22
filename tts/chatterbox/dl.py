"""Download the Chatterbox Multilingual (V2) weights from Hugging Face with retries.

`snapshot_download` stalled on the large files in our environment; if it does
the same for you, use fetch.sh (curl with resume) instead.

Usage:
    pip install huggingface_hub
    MODELS_DIR=./models python3 dl.py
"""
import os
import sys
import time

from huggingface_hub import snapshot_download

MODELS_DIR = os.environ.get("MODELS_DIR", "./models")
PATTERNS = ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
            "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]

for attempt in range(1, 21):
    try:
        path = snapshot_download(repo_id="ResembleAI/chatterbox", repo_type="model",
                                 revision="main", allow_patterns=PATTERNS,
                                 local_dir=MODELS_DIR, max_workers=1, tqdm_class=None)
        print("OK DIR:", path, flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(10)
else:
    print("GAVE UP", flush=True)
    sys.exit(1)
