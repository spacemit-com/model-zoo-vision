#!/bin/sh
# Download LightGlue models to ~/.cache/models/vision/lightglue/
# Remote: https://archive.spacemit.com/spacemit-ai/model_zoo/vision/lightglue/
# Assets (test images) via: bash scripts/download_assets.sh
# Run: sh examples/lightglue/scripts/download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/lightglue"
mkdir -p "$MODEL_DIR"

BASE_URL="${LIGHTGLUE_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/lightglue}"

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

download "lightglue_for_superpoint_512_depth1.fp16.onnx"
download "lightglue_for_superpoint_512_depth9.fp16.onnx"
echo "Done. Models in $MODEL_DIR"
echo "Default config uses: lightglue_for_superpoint_512_depth1.fp16.onnx"
