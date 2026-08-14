#!/bin/sh
# Download YOLOP model to ~/.cache/models/vision/yolop/.

set -e
MODEL_DIR="${HOME:-/tmp}/.cache/models/vision/yolop"
BASE_URL="${YOLOP_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolop}"
NAME="yolop-288-512-sim.q.onnx"
DESTINATION="$MODEL_DIR/$NAME"
IMAGE_DIR="${HOME:-/tmp}/.cache/assets/image"
IMAGE_NAME="020_yolop.jpg"
IMAGE_URL="${YOLOP_TEST_IMAGE_URL:-https://raw.githubusercontent.com/hustvl/YOLOP/main/test.jpg}"

mkdir -p "$MODEL_DIR" "$IMAGE_DIR"

download_one() {
  url="$1"
  destination="$2"
  if [ -s "$destination" ]; then
    echo "Exists: $destination"
    return 0
  fi
  part="$destination.part"
  rm -f "$part"
  echo "Downloading $(basename "$destination") ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$part" "$url"
  else
    echo "Error: need curl or wget." >&2
    return 1
  fi
  mv "$part" "$destination"
  echo "Done: $destination"
}

trap 'rm -f "$DESTINATION.part" "$IMAGE_DIR/$IMAGE_NAME.part"' EXIT HUP INT TERM
download_one "$IMAGE_URL" "$IMAGE_DIR/$IMAGE_NAME"
download_one "$BASE_URL/$NAME" "$DESTINATION"
