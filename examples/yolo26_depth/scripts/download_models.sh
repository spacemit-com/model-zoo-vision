#!/bin/sh
# Download model(s) for yolo26_depth example. Saved to ~/.cache/models/vision/yolo26_depth/

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/yolo26_depth"
mkdir -p "$MODEL_DIR"

BASE_URL="${BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo26_depth}"
download() {
  name="$1"
  if [ -f "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return 0
  fi
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$MODEL_DIR/$name.part" "$BASE_URL/$name"
  else
    wget -O "$MODEL_DIR/$name.part" "$BASE_URL/$name"
  fi
  mv "$MODEL_DIR/$name.part" "$MODEL_DIR/$name"
}

download "yolo26n-depth.fp16.onnx"
download "yolo26s-depth.fp16.onnx"
download "yolo26m-depth.fp16.onnx"
echo "Done. Models in $MODEL_DIR"
