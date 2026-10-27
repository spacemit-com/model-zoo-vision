#!/bin/sh
# Prepare YOLO26-Seg weights in ~/.cache/models/vision/yolo26_seg/.

set -eu
ROOT_DIR="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd)"
MODEL_DIR="${HOME:-/tmp}/.cache/models/vision/yolo26_seg"
BASE_URL="${YOLO26_SEG_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo26_seg}"
mkdir -p "$MODEL_DIR"

prepare_model() {
  name="$1"
  expected="$2"
  destination="$MODEL_DIR/$name"
  part_file="$destination.part"
  if [ -s "$destination" ]; then
    actual="$(sha256sum "$destination" | cut -d ' ' -f 1)"
    if [ "$actual" = "$expected" ]; then
      echo "Exists: $destination"
      return 0
    fi
    echo "Checksum mismatch: $destination" >&2
    return 1
  fi
  if [ -f "$ROOT_DIR/$name" ]; then
    echo "Copying $name from repository root ..."
    cp "$ROOT_DIR/$name" "$part_file"
  else
    echo "Downloading $name ..."
    if command -v curl >/dev/null 2>&1; then
      curl -fL -o "$part_file" "$BASE_URL/$name"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$part_file" "$BASE_URL/$name"
    else
      echo "Error: curl or wget is required." >&2
      return 1
    fi
  fi
  actual="$(sha256sum "$part_file" | cut -d ' ' -f 1)"
  if [ "$actual" != "$expected" ]; then
    echo "Checksum mismatch: $name" >&2
    return 1
  fi
  mv "$part_file" "$destination"
  echo "Ready: $destination"
}

prepare_model "yolo26n-seg.fp16.onnx" \
  "5fa1e5e598c3009fbe3bb83cbe2f77f12b23ef181fc775e31b25714faf6d0d1d"
prepare_model "yolo26s-seg.fp16.onnx" \
  "b1e01c6ae3d694671019f84495482293a2404d6050afd9694eaf44f7f000f4be"
prepare_model "yolo26m-seg.fp16.onnx" \
  "3f2daf7f9c64ccfd1848922c07277884dc7592a05d03e95ee2bddb4e49a99538"
