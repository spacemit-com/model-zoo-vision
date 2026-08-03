#!/bin/sh
# Download AVTrack models to ~/.cache/models/vision/avtrack/
# Remote: https://archive.spacemit.com/spacemit-ai/model_zoo/vision/AVTrack/
# Assets (test video) via: bash scripts/download_assets.sh
# Run: sh examples/avtrack/scripts/download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/avtrack"
mkdir -p "$MODEL_DIR"

BASE_URL="${AVTRACK_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/AVTrack}"

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

download "avtrack_deit_depth4.q.onnx"
download "avtrack_deit_depth6.q.onnx"
echo "Done. Models in $MODEL_DIR"
echo "Default config uses: avtrack_deit_depth4.q.onnx"
