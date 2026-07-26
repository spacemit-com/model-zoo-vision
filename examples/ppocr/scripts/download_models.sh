#!/bin/sh
# Download PP-OCRv6 models to ~/.cache/models/vision/ppocr/
# Remote: https://archive.spacemit.com/spacemit-ai/model_zoo/vision/ppocr/
# Run: bash examples/ppocr/scripts/download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/ppocr"
mkdir -p "$MODEL_DIR"

BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/ppocr"
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

# PP-OCRv6 tiny (default in config/ppocr.yaml)
download "PP-OCRv6_tiny_det_640x640.fp16.onnx"
download "PP-OCRv6_tiny_rec_48x320.dynq.onnx"

# PP-OCRv6 small
download "PP-OCRv6_small_det_640x640.fp16.onnx"
download "PP-OCRv6_small_rec_48x320.dynq.onnx"

echo "Done. Models in $MODEL_DIR"
echo "note: character dicts ship in the repo (no download):"
echo "      assets/labels/ppocrv6_tiny_dict.txt  (tiny rec)"
echo "      assets/labels/ppocrv6_small_dict.txt (small rec)"
