#!/usr/bin/env bash
# Download CosPlace test assets (Barcelona dataset) to ~/.cache/assets/image/barcelona/

set -euo pipefail

ASSET_DIR="${HOME}/.cache/assets/image"
BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/assets/image"
ASSET_FILE="021_barcelona.tar.xz"
ARCHIVE_ROOT="021_barcelona"
TARGET_DIR="${ASSET_DIR}/barcelona"

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

# The dataset is complete only if both query/ and database/ contain files.
is_complete() {
    [ -d "$1/query" ] && [ -d "$1/database" ] \
        && [ -n "$(ls -A "$1/query")" ] && [ -n "$(ls -A "$1/database")" ]
}

mkdir -p "${ASSET_DIR}"

if is_complete "${TARGET_DIR}"; then
    echo "Exists: ${TARGET_DIR}"
else
    # Download and extract into a private staging dir; only a fully extracted
    # and validated dataset is moved into place.
    STAGING_DIR="$(mktemp -d "${ASSET_DIR}/.barcelona.XXXXXX")"
    trap 'rm -rf "${STAGING_DIR}"' EXIT

    echo "Downloading ${ASSET_FILE} ..."
    download "${BASE_URL}/${ASSET_FILE}" "${STAGING_DIR}/${ASSET_FILE}"

    echo "Extracting ${ASSET_FILE} ..."
    tar -xJf "${STAGING_DIR}/${ASSET_FILE}" -C "${STAGING_DIR}"

    if ! is_complete "${STAGING_DIR}/${ARCHIVE_ROOT}"; then
        echo "Error: archive does not contain ${ARCHIVE_ROOT}/query and ${ARCHIVE_ROOT}/database" >&2
        exit 1
    fi

    rm -rf "${TARGET_DIR}"
    mv "${STAGING_DIR}/${ARCHIVE_ROOT}" "${TARGET_DIR}"
fi

echo "Done. Test data: ${TARGET_DIR}"
echo "  Query images:    ${TARGET_DIR}/query/"
echo "  Database images: ${TARGET_DIR}/database/"
