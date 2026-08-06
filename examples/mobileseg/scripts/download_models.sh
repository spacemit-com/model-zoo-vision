#!/bin/sh
# Prepare the MobileSeg model in ~/.cache/models/vision/mobileseg/.
# The repository-root model supplied for onboarding is preferred.
# Remote fallback:
# https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobileseg/

set -eu

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname "$0")" && pwd)"
REPOSITORY_ROOT="$(CDPATH='' cd -- "$SCRIPT_DIR/../../.." && pwd)"
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/mobileseg"
MODEL_NAME="mobileseg_mobilenetv2_cityscapes_1024x512.q.onnx"
EXPECTED_SHA256="98783c9b812336469aafac9131bfda94d1b3a1859a91385a0143d59077ef2744"
BASE_URL="${MOBILESEG_BASE_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobileseg}"
LOCAL_SOURCE="${MOBILESEG_MODEL_SOURCE:-$REPOSITORY_ROOT/$MODEL_NAME}"
DESTINATION="$MODEL_DIR/$MODEL_NAME"
PART_FILE="$DESTINATION.part"

mkdir -p "$MODEL_DIR"

cleanup() {
  rm -f "$PART_FILE"
}
trap cleanup EXIT HUP INT TERM

verify_checksum() {
  actual_sha256="$(sha256sum "$1" | awk '{print $1}')"
  if [ "$actual_sha256" != "$EXPECTED_SHA256" ]; then
    echo "Error: checksum mismatch for $1" >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Actual:   $actual_sha256" >&2
    return 1
  fi
}

if [ -s "$DESTINATION" ]; then
  verify_checksum "$DESTINATION"
  echo "Exists: $DESTINATION"
  exit 0
fi

rm -f "$PART_FILE"
if [ -f "$LOCAL_SOURCE" ]; then
  echo "Copying $MODEL_NAME from $LOCAL_SOURCE ..."
  cp "$LOCAL_SOURCE" "$PART_FILE"
else
  echo "Downloading $MODEL_NAME ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$PART_FILE" "$BASE_URL/$MODEL_NAME"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$PART_FILE" "$BASE_URL/$MODEL_NAME"
  else
    echo "Error: need curl or wget." >&2
    exit 1
  fi
fi

verify_checksum "$PART_FILE"
mv "$PART_FILE" "$DESTINATION"
echo "Done: $DESTINATION"
