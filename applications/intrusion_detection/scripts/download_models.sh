#!/bin/sh
# 区域闯入识别应用：下载模型到 ~/.cache/models/vision/
# 运行: sh download_models.sh（在 scripts 目录或应用根目录下执行）

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/yolov8"
mkdir -p "$MODEL_DIR"

BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8"
download() {
  name="$1"
  if [ -f "$MODEL_DIR/$name" ]; then
    echo "已存在: $MODEL_DIR/$name"
    return 0
  fi
  echo "下载 $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$MODEL_DIR/$name" "$BASE_URL/$name"
  else
    wget -O "$MODEL_DIR/$name" "$BASE_URL/$name"
  fi
}

download "yolov8n.q.onnx"

echo "完成。模型目录: $MODEL_DIR"
echo "配置中请使用: tracker_model_path: ~/.cache/models/vision/yolov8/yolov8n.q.onnx"
