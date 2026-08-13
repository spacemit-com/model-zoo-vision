#!/bin/sh
# Download MobileCLIP2-S3 image/text ONNX to ~/.cache/models/vision/mobileclip2/s3/
# BPE merges ship in the repo at assets/clip/bpe_merges.txt (no download).
set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/mobileclip2/s3"
mkdir -p "$MODEL_DIR"
BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobileclip2_s3"

download() {
  name="$1"
  if [ -f "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return 0
  fi
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$MODEL_DIR/$name" "$BASE_URL/$name"
  else
    wget -O "$MODEL_DIR/$name" "$BASE_URL/$name"
  fi
}

download "image_encoder.fp16.onnx"
download "text_encoder.onnx"
echo "Done. Models in $MODEL_DIR"
echo "Note: bpe_merges.txt ships in the repo at assets/clip/bpe_merges.txt (no download needed)."
