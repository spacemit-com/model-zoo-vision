#!/usr/bin/env bash
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Build (via mm) and run C++ vision_service.h PR tests.
# Usage: bash tests/pr/cpp/run_cpp_tests.sh functional|invalid

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "functional" && "$MODE" != "invalid" ]]; then
  echo "Usage: $0 functional|invalid" >&2
  exit 1
fi

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$MODULE_ROOT"

SDK_ROOT="${SROBOTIS_ROOT:-${SPACEMIT_SDK_ROOT:-}}"
STAGING="${SROBOTIS_OUTPUT_STAGING:-}"
if [[ -z "$STAGING" && -n "$SDK_ROOT" ]]; then
  STAGING="${SDK_ROOT}/output/staging"
fi

find_libvision() {
  local candidates=()
  if [[ -n "$STAGING" ]]; then
    candidates+=("${STAGING}/lib/libvision.so")
  fi
  candidates+=("${MODULE_ROOT}/build/libvision.so")
  for path in "${candidates[@]}"; do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

find_test_binary() {
  local name="$1"
  local candidates=()
  if [[ -n "$STAGING" ]]; then
    candidates+=("${STAGING}/bin/${name}")
  fi
  candidates+=("${MODULE_ROOT}/build/tests/${name}")
  for path in "${candidates[@]}"; do
    if [[ -x "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

setup_ld_library_path() {
  local libvision_path="$1"
  local lib_dir
  lib_dir="$(dirname "$libvision_path")"
  local paths=("$lib_dir")
  if [[ -d /opt/opencv-spacemit/lib ]]; then
    paths+=("/opt/opencv-spacemit/lib")
  fi
  if [[ -n "${SPACEMIT_DIR:-}" && -d "${SPACEMIT_DIR}/lib" ]]; then
    paths+=("${SPACEMIT_DIR}/lib")
  elif [[ -d /opt/spacemit/lib ]]; then
    paths+=("/opt/spacemit/lib")
  fi
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    paths+=("${LD_LIBRARY_PATH}")
  fi
  export LD_LIBRARY_PATH
  LD_LIBRARY_PATH="$(IFS=:; echo "${paths[*]}")"
}

ensure_model_and_image() {
  local model="${HOME}/.cache/models/vision/yolov8/yolov8n_no_dfl.q.onnx"
  local image="${HOME}/.cache/assets/image/006_test.jpg"
  local missing=0

  if [[ ! -f "$model" ]]; then
    echo "Missing model: $model"
    missing=1
  fi
  if [[ ! -f "$image" ]]; then
    echo "Missing image: $image"
    missing=1
  fi

  if [[ "$missing" -eq 1 ]]; then
    echo "Attempting fallback download (yolov8 model + referenced assets)..."
    bash examples/yolov8/scripts/download_models.sh
    bash scripts/download_assets.sh
  fi

  if [[ ! -f "$model" || ! -f "$image" ]]; then
    echo "ERROR: required model/image still missing after fallback download." >&2
    echo "  Model: $model" >&2
    echo "  Image: $image" >&2
    echo "Pre-seed PR runner cache or run:" >&2
    echo "  bash examples/yolov8/scripts/download_models.sh" >&2
    echo "  bash scripts/download_assets.sh" >&2
    exit 1
  fi

  IMAGE_PATH="$image"
}

LIBVISION="$(find_libvision || true)"
if [[ -z "$LIBVISION" ]]; then
  echo "ERROR: libvision.so not found." >&2
  echo "Build on board first: cd components/model_zoo/vision && mm" >&2
  exit 1
fi
setup_ld_library_path "$LIBVISION"

mkdir -p tests/output

if [[ "$MODE" == "functional" ]]; then
  BIN="$(find_test_binary vision_cpp_functional || true)"
  if [[ -z "$BIN" ]]; then
    echo "ERROR: vision_cpp_functional not found. Run mm with BUILD_TESTS=ON." >&2
    exit 1
  fi
  ensure_model_and_image
  CONFIG="examples/yolov8/config/yolov8.yaml"
  OUTPUT="tests/output/cpp_functional.txt"
  echo "Running functional test: $BIN"
  "$BIN" --config "$CONFIG" --image "$IMAGE_PATH" --output "$OUTPUT"
else
  BIN="$(find_test_binary vision_cpp_invalid_input || true)"
  if [[ -z "$BIN" ]]; then
    echo "ERROR: vision_cpp_invalid_input not found. Run mm with BUILD_TESTS=ON." >&2
    exit 1
  fi
  echo "Running invalid-input test: $BIN"
  "$BIN"
fi
