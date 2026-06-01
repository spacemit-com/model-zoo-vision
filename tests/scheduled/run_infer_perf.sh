#!/usr/bin/env bash
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Scheduled performance collection via vision_infer_benchmark.
# Optional threshold gate: set VISION_PERF_MIN_FPS to enable FPS assertion.

set -euo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$MODULE_ROOT"

CONFIG="${VISION_PERF_CONFIG:-examples/yolov8/config/yolov8.yaml}"
IMAGE="${VISION_PERF_IMAGE:-${HOME}/.cache/assets/image/006_test.jpg}"
RUNS="${VISION_PERF_RUNS:-50}"
WARMUP="${VISION_PERF_WARMUP:-5}"
OUTPUT="tests/output/infer_perf.txt"

SDK_ROOT="${SROBOTIS_ROOT:-${SPACEMIT_SDK_ROOT:-}}"
STAGING="${SROBOTIS_OUTPUT_STAGING:-}"
if [[ -z "$STAGING" && -n "$SDK_ROOT" ]]; then
  STAGING="${SDK_ROOT}/output/staging"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--config <yaml>] [--image <path>]" >&2
      exit 1
      ;;
  esac
done

ensure_model_and_image() {
  local model="${HOME}/.cache/models/vision/yolov8/yolov8n.q.onnx"
  local missing=0
  if [[ ! -f "$model" ]]; then
    echo "Missing model: $model"
    missing=1
  fi
  if [[ ! -f "$IMAGE" ]]; then
    echo "Missing image: $IMAGE"
    missing=1
  fi
  if [[ "$missing" -eq 1 ]]; then
    echo "Attempting fallback download..."
    bash examples/yolov8/scripts/download_models.sh
    bash scripts/download_assets.sh
  fi
  if [[ ! -f "$model" || ! -f "$IMAGE" ]]; then
    echo "ERROR: model or image still missing." >&2
    exit 1
  fi
}

find_benchmark() {
  local candidates=()
  if [[ -n "$STAGING" ]]; then
    candidates+=("${STAGING}/bin/vision_infer_benchmark")
  fi
  candidates+=("${MODULE_ROOT}/build/tests/benchmarks/vision_infer_benchmark")
  for path in "${candidates[@]}"; do
    if [[ -x "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  if command -v vision_infer_benchmark >/dev/null 2>&1; then
    command -v vision_infer_benchmark
    return 0
  fi
  return 1
}

setup_ld_library_path() {
  local paths=()
  if [[ -n "$STAGING" && -d "${STAGING}/lib" ]]; then
    paths+=("${STAGING}/lib")
  fi
  if [[ -d "${MODULE_ROOT}/build" ]]; then
    paths+=("${MODULE_ROOT}/build")
  fi
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

ensure_model_and_image

BENCH="$(find_benchmark || true)"
if [[ -z "$BENCH" ]]; then
  echo "ERROR: vision_infer_benchmark not found. Run mm with BUILD_TESTS=ON." >&2
  exit 1
fi

setup_ld_library_path
mkdir -p tests/output

echo "Running: $BENCH --config $CONFIG --image $IMAGE --runs $RUNS --warmup $WARMUP"
set +e
BENCH_OUTPUT="$("$BENCH" --config "$CONFIG" --image "$IMAGE" --runs "$RUNS" --warmup "$WARMUP" 2>&1)"
BENCH_RC=$?
set -e

echo "$BENCH_OUTPUT"
printf '%s\n' "$BENCH_OUTPUT" > "$OUTPUT"

if [[ "$BENCH_RC" -ne 0 ]]; then
  echo "ERROR: vision_infer_benchmark failed with exit code $BENCH_RC" >&2
  exit 1
fi

MODEL_INFER_MS="$(echo "$BENCH_OUTPUT" | awk -F': ' '/Avg model infer:/ {print $2; exit}' | tr -d ' ms')"
FPS="$(echo "$BENCH_OUTPUT" | awk -F': ' '/^FPS:/ {print $2; exit}')"

{
  echo "config=$CONFIG"
  echo "image=$IMAGE"
  echo "runs=$RUNS"
  echo "warmup=$WARMUP"
  echo "model_infer_ms=${MODEL_INFER_MS:-unknown}"
  echo "fps=${FPS:-unknown}"
} >> "$OUTPUT"

echo "Collected model_infer_ms=${MODEL_INFER_MS:-unknown} fps=${FPS:-unknown}"
echo "Report: $OUTPUT"

if [[ -n "${VISION_PERF_MIN_FPS:-}" ]]; then
  if [[ -z "${FPS:-}" ]]; then
    echo "ERROR: VISION_PERF_MIN_FPS is set but FPS could not be parsed." >&2
    exit 1
  fi
  awk -v fps="$FPS" -v min="${VISION_PERF_MIN_FPS}" 'BEGIN {
    if (fps + 0 < min + 0) {
      print "ERROR: FPS " fps " below threshold " min > "/dev/stderr";
      exit 1;
    }
  }'
  echo "FPS threshold check passed (>= ${VISION_PERF_MIN_FPS})"
fi
