#!/bin/sh
# Download PP-OCRv5 models to ~/.cache/models/vision/ppocr/
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

download "PP-OCRv5_mobile_det.onnx"
download "PP-OCRv5_mobile_rec.onnx"

echo "Done. Models in $MODEL_DIR"
echo "note: character dict ships in the repo at assets/labels/ppocr_keys.txt (no download)."
echo "      line 0 is 'blank' (CTC blank); it matches PP-OCRv5 rec output classes."
