#!/bin/sh
# YOLOE detector + text/VPE encoders. Old YOLOE segmentation uses another cache.
set -eu
MODEL_DIR="${HOME:-/tmp}/.cache/models/vision/yoloe_detect"
BASE_URL="${YOLOE_DETECT_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yoloe_detect}"
mkdir -p "$MODEL_DIR"
download() {
  name="$1"
  if [ -s "$MODEL_DIR/$name" ]; then
    echo "Exists: $MODEL_DIR/$name"
    return
  fi
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$MODEL_DIR/$name.part" "$BASE_URL/$name"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$MODEL_DIR/$name.part" "$BASE_URL/$name"
  else
    echo "Error: curl or wget is required." >&2
    return 1
  fi
  mv "$MODEL_DIR/$name.part" "$MODEL_DIR/$name"
}
download yoloe26n_text_sim.fp16.onnx
download yoloe26n_vpe_sim.fp16.onnx
download yoloe26n_vp_det_sim.fp16.onnx
echo "Done. Models in $MODEL_DIR"
