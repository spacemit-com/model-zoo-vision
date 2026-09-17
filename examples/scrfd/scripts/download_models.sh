#!/bin/sh
# Download only the SCRFD detector, sharing the buffalo_l model cache.
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
    echo "ERROR: download failed or empty: $name" >&2
    return 1
  fi
}

download "det_10g_fixed.q.onnx"
echo "Done. Models in $MODEL_DIR"
