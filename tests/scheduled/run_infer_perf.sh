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
# Requires Python 3 and PyYAML (also declared in tests/requirements.txt).

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

# First entry is the primary model; remaining entries are configured auxiliaries.
CONFIG_PATHS=()

load_config_paths() {
  # Parse once per config, preserving quoted paths, aliases and flow mappings.
  local cfg="$1"
  local parser_pid
  CONFIG_PATHS=()
  mapfile -d '' -t CONFIG_PATHS < <(python3 - "$cfg" <<'PY'
import os
import sys

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required; install it with python3 -m pip install PyYAML", file=sys.stderr)
    sys.exit(1)


def model_path(value, name):
    if value is None:
        return ""
    if not isinstance(value, str) or "\0" in value:
        raise ValueError(f"{name} must be a path string")
    return os.path.expanduser(value)


try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    params = config.get("default_params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("default_params must be a YAML mapping")
    paths = [model_path(config.get("model_path"), "model_path")]
    for key, value in params.items():
        if isinstance(key, str) and key.endswith("_path"):
            paths.append(model_path(value, f"default_params.{key}"))
    sys.stdout.buffer.write(b"".join(path.encode("utf-8") + b"\0" for path in paths))
except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
    print(f"ERROR: cannot read model paths from {sys.argv[1]}: {error}", file=sys.stderr)
    sys.exit(1)
PY
  )
  parser_pid=$!
  # Process-substitution failures do not propagate through mapfile itself.
  if ! wait "$parser_pid"; then
    CONFIG_PATHS=()
    return 1
  fi
}

sibling_weights() {
  # Keep alternate primary weights, but not explicitly configured auxiliaries.
  local model="$1"
  local dir
  dir="$(dirname "$model")"
  [[ -d "$dir" ]] || return 0
  local f auxiliary is_auxiliary
  for f in "$dir"/*.onnx; do
    [[ -e "$f" ]] || continue
    [[ "$f" -ef "$model" ]] && continue
    is_auxiliary=0
    for auxiliary in "${CONFIG_PATHS[@]:1}"; do
      if [[ "$f" -ef "$auxiliary" ]]; then
        is_auxiliary=1
        break
      fi
    done
    [[ "$is_auxiliary" -eq 1 ]] && continue
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
    effective_model="${CONFIG_PATHS[0]}"
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
  local cfg model status failed=0
  for cfg in examples/*/config/*.yaml; do
    [[ -e "$cfg" ]] || continue
    if ! load_config_paths "$cfg"; then
      printf '  %-42s -> ERROR (invalid config)\n' "$cfg"
      failed=1
      continue
    fi
    model="${CONFIG_PATHS[0]}"
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
  return "$failed"
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
  list_models
  exit $?
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
  load_config_paths "$CONFIG" || exit 1
  label="$(basename "$(dirname "$(dirname "$CONFIG")")")"
  run_one_model "$label" "$CONFIG" "$MODEL" "$IMAGE"
else
  # ---- default: iterate every config family (+ sibling weights) ----
  for cfg in examples/*/config/*.yaml; do
    [[ -e "$cfg" ]] || continue
    label="$(basename "$(dirname "$(dirname "$cfg")")")"
    if ! load_config_paths "$cfg"; then
      TOTAL=$((TOTAL + 1))
      FAILED=$((FAILED + 1))
      echo "[FAIL] $label: cannot read config ($cfg)"
      printf '%-22s | %-44s | %-10s | %s\n' "$label" "(invalid config)" "FAIL" "$cfg" >> "$SUMMARY"
      continue
    fi
    default_model="${CONFIG_PATHS[0]}"
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



