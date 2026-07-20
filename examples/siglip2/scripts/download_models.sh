#!/bin/sh
# Download SigLIP2 vision/text ONNX and tokenizer.bin to ~/.cache/models/vision/siglip2/
set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/siglip2"
mkdir -p "$MODEL_DIR"
BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/siglip2"

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

download "siglip2_vision_encoder_fp16_proj_dynq.onnx"
download "siglip2_text_encoder_dynq.onnx"
download "tokenizer.bin"
echo "Done. Models in $MODEL_DIR"
