#!/bin/sh
# Download RT-DETR-R18 and RT-DETRv2-S models to ~/.cache/models/vision/rtdetr/.

set -eu
MODEL_DIR="${HOME:-/tmp}/.cache/models/vision/rtdetr"
BASE_URL="${RTDETR_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/rtdetr}"

mkdir -p "$MODEL_DIR"

cleanup() {
  rm -f \
    "$MODEL_DIR/rtdetr_r18.fp16.onnx.part" \
    "$MODEL_DIR/rtdetrv2_s.fp16.onnx.part"
}
trap cleanup EXIT HUP INT TERM

verify_checksum() {
  actual_sha256="$(sha256sum "$1" | awk '{print $1}')"
  if [ "$actual_sha256" != "$2" ]; then
    echo "Error: checksum mismatch for $1" >&2
    echo "Expected: $2" >&2
    echo "Actual:   $actual_sha256" >&2
    return 1
  fi
}

download() {
  name="$1"
  expected_sha256="$2"
  destination="$MODEL_DIR/$name"
  part_file="$destination.part"

  if [ -s "$destination" ]; then
    verify_checksum "$destination" "$expected_sha256"
    echo "Exists: $destination"
    return 0
  fi

  rm -f "$part_file"
  echo "Downloading $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$part_file" "$BASE_URL/$name"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$part_file" "$BASE_URL/$name"
  else
    echo "Error: need curl or wget." >&2
    return 1
  fi
  verify_checksum "$part_file" "$expected_sha256"
  mv "$part_file" "$destination"
  echo "Done: $destination"
}

download "rtdetr_r18.fp16.onnx" \
  "eb0b478b8aadef57c176140b5c12d0bfd3d584c89bea123ba60238a76878dd8c"
download "rtdetrv2_s.fp16.onnx" \
  "9c6b6b747a703329789823b1bb0baf0e9cb0012dbadb8c73d92e77771eecdeb9"
echo "Done. Models in $MODEL_DIR"
