#!/bin/sh
# Download model(s) for yolo26 example. Saved to ~/.cache/models/vision/yolo26/
# Download URLs for this example. Run with: sh download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/yolo26"
mkdir -p "$MODEL_DIR"

BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo26"
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


download "yolo26n.q.onnx"
download "yolo26s.q.onnx"
download "yolo26m.q.onnx"
echo "Done. Models in $MODEL_DIR"
