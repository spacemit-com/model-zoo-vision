#!/bin/sh
# Prepare DEIMv2 N/S/M models in ~/.cache/models/vision/deimv2/.
# Repository-root models supplied for onboarding are preferred.

set -eu

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="$(CDPATH='' cd -- "$SCRIPT_DIR/../../.." && pwd)"
MODEL_DIR="${HOME:-/tmp}/.cache/models/vision/deimv2"
BASE_URL="${DEIMV2_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/deimv2}"
LOCAL_MODEL_DIR="${DEIMV2_MODEL_DIR:-$REPOSITORY_ROOT}"

mkdir -p "$MODEL_DIR"

cleanup() {
  rm -f \
    "$MODEL_DIR/deimv2n.fp16.onnx.part" \
    "$MODEL_DIR/deimv2s.fp16.onnx.part" \
    "$MODEL_DIR/deimv2m.fp16.onnx.part"
}
trap cleanup EXIT HUP INT TERM

verify_checksum() {
  expected_sha256="$2"
  actual_sha256="$(sha256sum "$1" | awk '{print $1}')"
  if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "Error: checksum mismatch for $1" >&2
    echo "Expected: $expected_sha256" >&2
    echo "Actual:   $actual_sha256" >&2
    return 1
  fi
}

prepare_model() {
  model_name="$1"
  expected_sha256="$2"
  destination="$MODEL_DIR/$model_name"
  part_file="$destination.part"
  local_source="$LOCAL_MODEL_DIR/$model_name"

  if [ -s "$destination" ]; then
    verify_checksum "$destination" "$expected_sha256"
    echo "Exists: $destination"
    return 0
  fi

  rm -f "$part_file"
  if [ -f "$local_source" ]; then
    echo "Copying $model_name from $local_source ..."
    cp "$local_source" "$part_file"
  else
    echo "Downloading $model_name ..."
    if command -v curl >/dev/null 2>&1; then
      curl -fL -o "$part_file" "$BASE_URL/$model_name"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$part_file" "$BASE_URL/$model_name"
    else
      echo "Error: need curl or wget." >&2
      return 1
    fi
  fi

  verify_checksum "$part_file" "$expected_sha256"
  mv "$part_file" "$destination"
  echo "Done: $destination"
}

prepare_model "deimv2n.fp16.onnx" \
  "8084038192edda8fa9cb517b75ddf0109c681592bd0664a7badeffcd4eb10ad6"
prepare_model "deimv2s.fp16.onnx" \
  "589a1e35e05667fc44f694882c80d20df82af19780075d2a7a4a619164372e11"
prepare_model "deimv2m.fp16.onnx" \
  "fae253df8cdd780721bc055bd843890079c8b29b968aa146801cb4b833a8ed18"
