#!/bin/sh
# Download YOLOE models to ~/.cache/models/vision/yoloe/
# Remote: https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yoloe/
# Run: bash examples/yoloe/scripts/download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/yoloe"
mkdir -p "$MODEL_DIR"

BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yoloe"
download() {
  name="$1"
  if [ -f "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return 0
  fi
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$MODEL_DIR/$name" "$BASE_URL/$name"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$MODEL_DIR/$name" "$BASE_URL/$name"
  else
    echo "Error: need curl or wget."
    exit 1
  fi
}

download "yoloe-v8s-seg.dynq.onnx"
download "mobileclip.q.onnx"

echo "Done. Models in $MODEL_DIR"
echo "Note: bpe_merges.txt ships in the repo at assets/clip/bpe_merges.txt (no download needed)."
