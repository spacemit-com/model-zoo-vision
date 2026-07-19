#!/bin/sh
# Download buffalo_l models to ~/.cache/models/vision/buffalo_l/
set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/buffalo_l"
mkdir -p "$MODEL_DIR"
BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/buffalo_l"

download() {
  name="$1"
  if [ -s "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return 0
  fi
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fSL -o "$MODEL_DIR/$name" "$BASE_URL/$name"
  else
    wget -O "$MODEL_DIR/$name" "$BASE_URL/$name"
  fi
  if [ ! -s "$MODEL_DIR/$name" ]; then
    rm -f "$MODEL_DIR/$name"
    echo "ERROR: download failed or empty: $name ($BASE_URL/$name)" >&2
    return 1
  fi
}

download "det_10g.q.onnx"
download "det_10g_fixed.q.onnx"
download "w600k_r50.q.onnx"
download "genderage.q.onnx"
download "2d106det.onnx"

echo "Done. Models in $MODEL_DIR"
