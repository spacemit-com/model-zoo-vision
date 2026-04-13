#!/bin/sh
# Download model(s) for yolov5 example. Saved to ~/.cache/models/vision/yolov5/

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/yolov5"
mkdir -p "$MODEL_DIR"

download() {
  url="$1"
  name="$2"
  if [ -f "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return 0
  fi
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$MODEL_DIR/$name" "$url"
  else
    wget -O "$MODEL_DIR/$name" "$url"
  fi
}

download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5n.q.onnx" "yolov5n.q.onnx"
download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5/yolov5s.q.onnx" "yolov5s.q.onnx"
echo "Done. Models in $MODEL_DIR"
