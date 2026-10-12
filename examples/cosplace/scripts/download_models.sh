#!/usr/bin/env bash
# Download CosPlace models

set -e

MODEL_DIR="${HOME}/.cache/models/vision/cosplace"
BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/cosplace"

mkdir -p "${MODEL_DIR}"

echo "Downloading CosPlace models..."

# Download ResNet50 512-dim quantized model
MODEL_FILE="cosplace_resnet50_512.q.onnx"
if [ -f "${MODEL_DIR}/${MODEL_FILE}" ]; then
    echo "  ${MODEL_FILE} already exists, skipping."
else
    echo "  Downloading ${MODEL_FILE}..."
    wget -q --show-progress -O "${MODEL_DIR}/${MODEL_FILE}" \
        "${BASE_URL}/${MODEL_FILE}"
    echo "  ${MODEL_FILE} downloaded."
fi

echo "All CosPlace models downloaded successfully."
echo "Model location: ${MODEL_DIR}/${MODEL_FILE}"
