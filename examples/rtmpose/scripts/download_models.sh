#!/bin/sh
# Download RTMPose FP16 models to ~/.cache/models/vision/rtmpose/.

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="${MODEL_DIR:-$CACHE_BASE/rtmpose}"
mkdir -p "$MODEL_DIR"

BASE_URL="${BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/rtmpose}"
download() {
  name="$1"
  if [ -s "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return 0
  fi
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$MODEL_DIR/$name.part" "${BASE_URL%/}/$name"
  else
    wget -O "$MODEL_DIR/$name.part" "${BASE_URL%/}/$name"
  fi
  mv "$MODEL_DIR/$name.part" "$MODEL_DIR/$name"
}

download "rtmpose_s.fp16.onnx"
download "rtmpose_m.fp16.onnx"
echo "Done. Models in $MODEL_DIR"
