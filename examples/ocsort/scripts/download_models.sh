#!/bin/sh
# Download model(s) for ocsort example. Saved to ~/.cache/models/vision/ocsort/
# Download URLs for this example. Run with: sh download_models.sh

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/ocsort"
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

# Detector used by ocsort config
download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8/yolov8n.q.onnx" "yolov8n.q.onnx"
# Optional: OC-SORT YOLOX backbone (if config uses it)
download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/ocsort/ocsort_yoloxs_sim.onnx" "ocsort_yoloxs_sim.onnx"
echo "Done. Models in $MODEL_DIR"
