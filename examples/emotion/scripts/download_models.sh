#!/bin/sh
# Download model(s) for emotion example. Saved to ~/.cache/models/vision/emotion/
# Run with: sh download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/emotion"
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

download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/emotion/emotion_resnet50_final.q.onnx" "emotion_resnet50_final.q.onnx"
echo "Done. Models in $MODEL_DIR"
