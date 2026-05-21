#!/bin/sh
# Download model(s) for pp_liteseg example. Saved to ~/.cache/models/vision/pp_liteseg/
# Download URLs for this example. Run with: sh download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/pp_liteseg"
mkdir -p "$MODEL_DIR"

BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/pp_liteseg"
download() {
  name="$1"
  if [ -f "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return 0
  fi
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$MODEL_DIR/$name" "$BASE_URL/$name"
  else
    wget -O "$MODEL_DIR/$name" "$BASE_URL/$name"
  fi
}

download "pp_liteseg.q.onnx"
echo "Done. Models in $MODEL_DIR"
