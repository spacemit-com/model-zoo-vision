#!/usr/bin/env bash
# Download CosPlace test assets (Barcelona dataset)

set -e

ASSET_DIR="${HOME}/.cache/assets/image"
BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/assets/image"

mkdir -p "${ASSET_DIR}"

echo "Downloading CosPlace test assets..."

# Download Barcelona test images
ASSET_FILE="021_barcelona.tar.xz"
ASSET_EXTRACT_DIR="${ASSET_DIR}/barcelona"
if [ -d "${ASSET_EXTRACT_DIR}" ]; then
    echo "  Barcelona test images already exist, skipping."
else
    echo "  Downloading Barcelona test images..."
    wget -q --show-progress -O "/tmp/${ASSET_FILE}" \
        "${BASE_URL}/${ASSET_FILE}"
    echo "  Extracting Barcelona test images..."
    tar -xJf "/tmp/${ASSET_FILE}" -C "${ASSET_DIR}"
    mv "${ASSET_DIR}/021_barcelona" "${ASSET_EXTRACT_DIR}"
    rm "/tmp/${ASSET_FILE}"
    echo "  Barcelona test images extracted."
fi

echo "All CosPlace test assets downloaded successfully."
echo "Test data location: ${ASSET_EXTRACT_DIR}"
echo "  Query images: ${ASSET_EXTRACT_DIR}/query/"
echo "  Database images: ${ASSET_EXTRACT_DIR}/database/"
