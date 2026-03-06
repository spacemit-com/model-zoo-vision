#!/bin/sh
# Download model(s) for emotion example. Saved to ~/.cache/models/vision/resnet/
# Run with: sh download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/resnet"
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

# 情绪模型存放在 vision/resnet 目录
download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/resnet/emotion_resnet50_final.q.onnx" "emotion_resnet50_final.q.onnx"
echo "Done. Models in $MODEL_DIR"
