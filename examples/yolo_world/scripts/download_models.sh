#!/bin/sh
# Download YOLO-World models to ~/.cache/models/vision/yolo_world/
# Remote: https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo_world/
# Run: bash examples/yolo_world/scripts/download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/yolo_world"
mkdir -p "$MODEL_DIR"

BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo_world"
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

download "yolov8s-worldv2.dynq.onnx"
download "clip_text.onnx"

echo "Done. Models in $MODEL_DIR"
echo "Note: bpe_merges.txt ships in the repo at assets/clip/bpe_merges.txt (no download needed)."
