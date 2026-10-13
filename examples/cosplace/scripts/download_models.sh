#!/usr/bin/env bash
# Download CosPlace models to ~/.cache/models/vision/cosplace/

set -euo pipefail

MODEL_DIR="${HOME}/.cache/models/vision/cosplace"
BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/cosplace"
MODEL_FILE="cosplace_resnet50_512.q.onnx"

TARGET="${MODEL_DIR}/${MODEL_FILE}"
TMP_FILE="${TARGET}.tmp"

cleanup() {
    rm -f "${TMP_FILE}"
}
trap cleanup EXIT

download() {
    url="$1"
    out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "${out}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "${out}" "${url}"
    else
        echo "Error: neither curl nor wget is available." >&2
        exit 1
    fi
}

mkdir -p "${MODEL_DIR}"

# A non-empty file is treated as complete; the download goes to a .tmp file
# and is only renamed on success, so an interrupted run never leaves a
# partial model at the final path.
if [ -s "${TARGET}" ]; then
    echo "Exists: ${TARGET}"
else
    rm -f "${TARGET}"
    echo "Downloading ${MODEL_FILE} ..."
    download "${BASE_URL}/${MODEL_FILE}" "${TMP_FILE}"
    if [ ! -s "${TMP_FILE}" ]; then
        echo "Error: downloaded file is empty: ${MODEL_FILE}" >&2
        exit 1
    fi
    mv "${TMP_FILE}" "${TARGET}"
fi

echo "Done. Model: ${TARGET}"
