#!/bin/sh
# Download SuperPoint models to ~/.cache/models/vision/superpoint/
# Remote: https://archive.spacemit.com/spacemit-ai/model_zoo/vision/superpoint/
# Assets (test image) via: bash scripts/download_assets.sh
# Run: sh examples/superpoint/scripts/download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/superpoint"
mkdir -p "$MODEL_DIR"

BASE_URL="${SUPERPOINT_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/superpoint}"

download() {
  name="$1"
  if [ -f "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return 0
  fi
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$MODEL_DIR/$name" "$BASE_URL/$name"
  else
    wget -O "$MODEL_DIR/$name" "$BASE_URL/$name"
  fi
}

download "superpoint_512x512_top512_batch1.q.onnx"
download "superpoint_512x512_top512_batch1.onnx"
echo "Done. Models in $MODEL_DIR"
echo "Default config uses: superpoint_512x512_top512_batch1.q.onnx"
