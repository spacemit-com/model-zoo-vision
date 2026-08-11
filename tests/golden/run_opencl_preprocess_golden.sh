#!/usr/bin/env bash
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vision-opencl-golden.XXXXXX")"

cleanup() {
  rm -rf -- "$BUILD_DIR"
}
trap cleanup EXIT

cmake -S "$MODULE_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DBUILD_PYTHON_WHEEL=OFF \
  -DBUILD_TESTS=ON \
  -DVISION_WITH_OPENCL=ON

cmake --build "$BUILD_DIR" \
  --target vision_opencl_image_preprocess_golden_test \
  -j "${VISION_BUILD_JOBS:-4}"

"$BUILD_DIR/tests/vision_opencl_image_preprocess_golden_test"
