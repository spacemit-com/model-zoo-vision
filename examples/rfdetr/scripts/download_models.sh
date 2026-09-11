#!/bin/sh
# Download RF-DETR Nano/Small/Medium models to ~/.cache/models/vision/rfdetr/.

set -e
SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="$(CDPATH='' cd -- "$SCRIPT_DIR/../../.." && pwd)"
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/rfdetr"
mkdir -p "$MODEL_DIR"

BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/rfdetr"
download() {
  name="$1"
  if [ -f "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return 0
  fi
  if [ -f "$REPOSITORY_ROOT/$name" ]; then
    echo "Copying $name from repository root ..."
    cp "$REPOSITORY_ROOT/$name" "$MODEL_DIR/$name"
  elif command -v curl >/dev/null 2>&1; then
    echo "Downloading $name ..."
    curl -L -o "$MODEL_DIR/$name" "$BASE_URL/$name"
  else
    echo "Downloading $name ..."
    wget -O "$MODEL_DIR/$name" "$BASE_URL/$name"
  fi
}

download "rfdetr-nano.fp16.onnx"
download "rfdetr-small.fp16.onnx"
download "rfdetr-medium.fp16.onnx"
echo "Done. Models in $MODEL_DIR"
