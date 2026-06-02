#!/usr/bin/env bash
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Scheduled performance collection via vision_infer_benchmark.
#
# Default behavior: iterate every examples/*/config/*.yaml, run the benchmark
# for each model family using the image/video declared in that config, and
# additionally test sibling weights of the same family (e.g. yolov8n/s/m).
# Missing models are reported and skipped, not failed.
#
# Single-model mode: pass --config (and optionally --model-path/--image) or set
# VISION_PERF_CONFIG to benchmark exactly one model.
#
# Optional FPS gate: set VISION_PERF_MIN_FPS to assert every measured model
# meets the threshold.

set -uo pipefail

MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$MODULE_ROOT" || exit

RUNS="${VISION_PERF_RUNS:-50}"
WARMUP="${VISION_PERF_WARMUP:-5}"
OUTPUT="tests/output/infer_perf.txt"

# Single-model overrides (empty = iterate all configs).
CONFIG="${VISION_PERF_CONFIG:-}"
MODEL="${VISION_PERF_MODEL:-}"
IMAGE="${VISION_PERF_IMAGE:-}"
LIST_ONLY=0

SDK_ROOT="${SROBOTIS_ROOT:-${SPACEMIT_SDK_ROOT:-}}"
STAGING="${SROBOTIS_OUTPUT_STAGING:-}"
if [[ -z "$STAGING" && -n "$SDK_ROOT" ]]; then
  STAGING="${SDK_ROOT}/output/staging"
fi

usage() {
  cat <<'EOF'
Usage: run_infer_perf.sh [options]

Default (no --config): benchmark every model family under examples/*/config/*.yaml,
plus sibling weights of the same family. Missing models are skipped with a note.

Options:
  --config <yaml>      Benchmark only this config (single-model mode).
  --model-path <onnx>  Override the model weight (use with --config).
  --image <path>       Override the input image/video (default: config's own).
  --list-models        List configs and whether their models are present, then exit.
  --help               Show this help.

Environment variables:
  VISION_PERF_CONFIG   Same as --config.
  VISION_PERF_MODEL    Same as --model-path.
  VISION_PERF_IMAGE    Same as --image.
  VISION_PERF_RUNS     Timing iterations (default 50).
  VISION_PERF_WARMUP   Warmup iterations (default 5).
  VISION_PERF_MIN_FPS  If set, every measured model must reach this FPS or the run fails.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      CONFIG="$2"; shift 2 ;;
    --model-path)  MODEL="$2"; shift 2 ;;
    --image)       IMAGE="$2"; shift 2 ;;
    --list-models) LIST_ONLY=1; shift ;;
    --help|-h)     usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

expand_tilde() {
  # Expand a leading ~ to $HOME.
  local p="$1"
  if [[ "$p" == "~"* ]]; then
    printf '%s' "${HOME}${p:1}"
  else
    printf '%s' "$p"
  fi
}

config_model_path() {
  # Extract the first model_path: from a config yaml, tilde-expanded.
  local cfg="$1"
  local raw
  raw="$(grep -m1 -E '^[[:space:]]*model_path:' "$cfg" | sed -E 's/^[[:space:]]*model_path:[[:space:]]*//; s/[[:space:]]*(#.*)?$//')"
  [[ -z "$raw" ]] && return 1
  expand_tilde "$raw"
}

sibling_weights() {
  # Given a model file, echo other *.onnx in the same dir (same family).
  local model="$1"
  local dir
  dir="$(dirname "$model")"
  [[ -d "$dir" ]] || return 0
  local f
  for f in "$dir"/*.onnx; do
    [[ -e "$f" ]] || continue
    [[ "$f" == "$model" ]] && continue
    echo "$f"
  done
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

# Counters and gate state (shared by run_one_model).
TOTAL=0
MEASURED=0
SKIPPED=0
FAILED=0
GATE_FAILED=0

run_one_model() {
  # Args: <label> <config> <model_path_or_empty> <image_or_empty>
  # Runs benchmark, parses metrics, appends a summary row, applies optional gate.
  local label="$1" cfg="$2" model="$3" img="$4"
  TOTAL=$((TOTAL + 1))

  # Resolve the model that will actually be used (override or config default).
  local effective_model="$model"
  if [[ -z "$effective_model" ]]; then
    effective_model="$(config_model_path "$cfg" || true)"
  fi

  if [[ -z "$effective_model" ]]; then
    echo "[SKIP] $label: config has no model_path ($cfg)"
    printf '%-22s | %-44s | %-10s | %s\n' "$label" "(no model_path)" "SKIP" "$cfg" >> "$SUMMARY"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi
  if [[ ! -f "$effective_model" ]]; then
    echo "[SKIP] $label: model not downloaded -> $effective_model"
    echo "        get it via: bash examples/<model>/scripts/download_models.sh"
    printf '%-22s | %-44s | %-10s | %s\n' "$label" "$(basename "$effective_model")" "SKIP" "missing" >> "$SUMMARY"
    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  local cmd=("$BENCH" --config "$cfg" --runs "$RUNS" --warmup "$WARMUP")
  [[ -n "$model" ]] && cmd+=(--model-path "$model")
  [[ -n "$img" ]] && cmd+=(--image "$img")

  echo "[RUN ] $label: ${cmd[*]}"
  local out rc
  out="$("${cmd[@]}" 2>&1)"
  rc=$?
  printf '\n===== %s =====\n%s\n' "$label" "$out" >> "$RAW_LOG"

  if [[ "$rc" -ne 0 ]]; then
    echo "[FAIL] $label: benchmark exit code $rc (see $RAW_LOG)"
    printf '%-22s | %-44s | %-10s | rc=%s\n' "$label" "$(basename "$effective_model")" "FAIL" "$rc" >> "$SUMMARY"
    FAILED=$((FAILED + 1))
    return 0
  fi

  local ms fps
  ms="$(echo "$out" | awk -F': ' '/Avg model infer:/ {print $2; exit}' | tr -d ' ms')"
  fps="$(echo "$out" | awk -F': ' '/^FPS:/ {print $2; exit}')"
  echo "[ OK ] $label: model_infer_ms=${ms:-?} fps=${fps:-?}"
  printf '%-22s | %-44s | ms=%-8s | fps=%s\n' \
    "$label" "$(basename "$effective_model")" "${ms:-?}" "${fps:-?}" >> "$SUMMARY"
  MEASURED=$((MEASURED + 1))

  if [[ -n "${VISION_PERF_MIN_FPS:-}" ]]; then
    if [[ -z "${fps:-}" ]]; then
      echo "[GATE] $label: FPS unparseable but VISION_PERF_MIN_FPS set" >&2
      GATE_FAILED=$((GATE_FAILED + 1))
    elif awk -v f="$fps" -v m="$VISION_PERF_MIN_FPS" 'BEGIN { exit !(f + 0 < m + 0) }'; then
      echo "[GATE] $label: FPS $fps below threshold $VISION_PERF_MIN_FPS" >&2
      GATE_FAILED=$((GATE_FAILED + 1))
    fi
  fi
}

# ---- list-models: show every config and whether its model is present ----
list_models() {
  echo "Configs under examples/*/config/*.yaml and model availability:"
  echo "------------------------------------------------------------------"
  local cfg model status
  for cfg in examples/*/config/*.yaml; do
    [[ -e "$cfg" ]] || continue
    model="$(config_model_path "$cfg" || true)"
    if [[ -z "$model" ]]; then
      status="NO model_path"
    elif [[ -f "$model" ]]; then
      status="present"
    else
      status="MISSING"
    fi
    printf '  %-42s -> %-9s %s\n' "$cfg" "$status" "${model/#$HOME/~}"
    # A+ : list sibling weights present in the same family dir
    if [[ -n "$model" ]]; then
      local sib
      while IFS= read -r sib; do
        [[ -z "$sib" ]] && continue
        printf '  %-42s    + sibling %s\n' "" "${sib/#$HOME/~}"
      done < <(sibling_weights "$model")
    fi
  done
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
  list_models
  exit 0
fi

# ---- locate benchmark + libs ----
BENCH="$(find_benchmark || true)"
if [[ -z "$BENCH" ]]; then
  echo "ERROR: vision_infer_benchmark not found. Run mm with BUILD_TESTS=ON." >&2
  exit 1
fi
setup_ld_library_path

mkdir -p tests/output
SUMMARY="$OUTPUT"
RAW_LOG="tests/output/infer_perf_raw.log"
: > "$SUMMARY"
: > "$RAW_LOG"
{
  echo "# vision inference performance summary"
  echo "# runs=$RUNS warmup=$WARMUP min_fps=${VISION_PERF_MIN_FPS:-<unset>}"
  echo "# label                | model                                        | metrics"
  echo "# ---------------------------------------------------------------------------"
} >> "$SUMMARY"

# ---- single-model mode (explicit --config) ----
if [[ -n "$CONFIG" ]]; then
  label="$(basename "$(dirname "$(dirname "$CONFIG")")")"
  run_one_model "$label" "$CONFIG" "$MODEL" "$IMAGE"
else
  # ---- default: iterate every config family (+ sibling weights) ----
  for cfg in examples/*/config/*.yaml; do
    [[ -e "$cfg" ]] || continue
    label="$(basename "$(dirname "$(dirname "$cfg")")")"
    default_model="$(config_model_path "$cfg" || true)"
    # Primary weight (config default).
    run_one_model "$label" "$cfg" "" "$IMAGE"
    # A+ sibling weights of the same family (only if default model exists).
    if [[ -n "$default_model" && -f "$default_model" ]]; then
      while IFS= read -r sib; do
        [[ -z "$sib" ]] && continue
        run_one_model "${label}:$(basename "$sib" .q.onnx)" "$cfg" "$sib" "$IMAGE"
      done < <(sibling_weights "$default_model")
    fi
  done
fi

# ---- footer + exit status ----
{
  echo "# ---------------------------------------------------------------------------"
  echo "# total=$TOTAL measured=$MEASURED skipped=$SKIPPED failed=$FAILED gate_failed=$GATE_FAILED"
} >> "$SUMMARY"

echo ""
echo "===== Performance summary ($SUMMARY) ====="
cat "$SUMMARY"
echo "Raw benchmark output: $RAW_LOG"

if [[ "$MEASURED" -eq 0 ]]; then
  echo "ERROR: no model was measured (all skipped/failed). Download models first or check build." >&2
  exit 1
fi
if [[ "$FAILED" -gt 0 ]]; then
  echo "ERROR: $FAILED model(s) failed to benchmark." >&2
  exit 1
fi
if [[ "$GATE_FAILED" -gt 0 ]]; then
  echo "ERROR: $GATE_FAILED model(s) below VISION_PERF_MIN_FPS=${VISION_PERF_MIN_FPS}." >&2
  exit 1
fi
echo "OK: measured=$MEASURED skipped=$SKIPPED"



