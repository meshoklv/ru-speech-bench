#!/bin/bash
# Download the Chatterbox Multilingual (V2) weights with curl, resuming partial files.
# Usage: MODELS_DIR=./models ./fetch.sh
set -u
MODELS_DIR="${MODELS_DIR:-./models}"
mkdir -p "$MODELS_DIR" && cd "$MODELS_DIR" || exit 1
B=https://huggingface.co/ResembleAI/chatterbox/resolve/main
for f in s3gen.pt t3_mtl23ls_v2.safetensors ve.pt grapheme_mtl_merged_expanded_v1.json conds.pt Cangjie5_TC.json; do
  for try in $(seq 1 40); do
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    curl -sL --max-time 600 --connect-timeout 30 -C - -o "$f" "$B/$f" && { echo "DONE $f $(stat -c%s "$f")"; break; }
    new=$(stat -c%s "$f" 2>/dev/null || echo 0)
    echo "retry $try $f at $new bytes"
    [ "$new" = "$sz" ] && sleep 15
  done
done
echo "ALL FINISHED"
