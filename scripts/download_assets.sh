#!/bin/sh
# 仅下载 model_zoo 的 image/、video/ 到 ~/.cache/assets（不下载 audio 等）
# 远程：https://archive.spacemit.com/spacemit-ai/model_zoo/assets/
# 在 cv 组件根目录执行: bash scripts/download_assets.sh

set -e
BASE_URL="${VISION_ASSETS_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/assets}"
CACHE="${HOME}/.cache/assets"
parent="$(dirname "$CACHE")"
mkdir -p "$CACHE"

echo "========== Download assets (image, video only) to $CACHE =========="
echo "Remote: $BASE_URL"

if command -v wget >/dev/null 2>&1; then
    for sub in image video; do
        echo ">>> $sub/"
        wget -q --show-progress -r -nH -np --cut-dirs=2 -R "index.html*" \
            -P "$parent" \
            "$BASE_URL/$sub/" || true
        if [ -d "$parent/spacemit-ai/model_zoo/assets/$sub" ]; then
            mkdir -p "$CACHE/$sub"
            cp -an "$parent/spacemit-ai/model_zoo/assets/$sub/"* "$CACHE/$sub/" 2>/dev/null || true
        fi
    done
    rm -rf "$parent/spacemit-ai" 2>/dev/null || true
elif command -v curl >/dev/null 2>&1; then
    echo "Curl: recursive fetch not supported. Create $CACHE/image and $CACHE/video and add URLs to download."
    for sub in image video; do mkdir -p "$CACHE/$sub"; done
else
    echo "Error: need wget. Install wget for full download."
    exit 1
fi

echo ""
echo "========== Done =========="
echo "Assets cache: $CACHE"
ls -la "$CACHE" 2>/dev/null || true
