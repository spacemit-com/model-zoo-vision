#!/bin/sh
# Download NanoTrack models to ~/.cache/models/vision/nanotrack/
# Remote: https://archive.spacemit.com/spacemit-ai/model_zoo/vision/nanotrack/
# Assets (test video) via: bash scripts/download_assets.sh
# Run: sh examples/nanotrack/scripts/download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/nanotrack"
BASE_URL="${NANOTRACK_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/nanotrack}"
CURRENT_PART=""

mkdir -p "$MODEL_DIR"

cleanup() {
  if [ -n "$CURRENT_PART" ]; then
    rm -f "$CURRENT_PART"
  fi
}
trap cleanup EXIT HUP INT TERM

download() {
  name="$1"
  destination="$MODEL_DIR/$name"
  if [ -s "$destination" ]; then
    echo "Exists: $destination"
    return 0
  fi
  CURRENT_PART="$destination.part"
  rm -f "$CURRENT_PART"
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$CURRENT_PART" "$BASE_URL/$name"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$CURRENT_PART" "$BASE_URL/$name"
  else
    echo "Error: need curl or wget." >&2
    return 1
  fi
  mv "$CURRENT_PART" "$destination"
  CURRENT_PART=""
}

download "nanotrack_backbone1.onnx"
download "nanotrack_backbone2.q.onnx"
download "nanotrack_head.q.onnx"
echo "Done. Models in $MODEL_DIR"
echo "Default config uses all three NanoTrack model files."
