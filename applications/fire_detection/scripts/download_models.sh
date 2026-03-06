#!/bin/sh
# 火焰检测应用：下载模型到 ~/.cache/models/vision/
# 运行: sh download_models.sh（在 scripts 目录或应用根目录下执行）

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE/yolov8"
mkdir -p "$MODEL_DIR"

BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8"
# 服务器目录中文件名为 yolov8%5Ffire.q.onnx（下划线 URL 编码）时使用；若实际文件名为 yolov8_fire.q.onnx 则改为 MODEL_URL="$BASE_URL/yolov8_fire.q.onnx"
MODEL_URL="$BASE_URL/yolov8%5Ffire.q.onnx"
MODEL_NAME="yolov8_fire.q.onnx"

if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
  echo "已存在: $MODEL_DIR/$MODEL_NAME"
else
  echo "下载 $MODEL_NAME ..."
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
  else
    wget -O "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
  fi
fi

echo "完成。模型目录: $MODEL_DIR"
echo "配置中请使用: detector_model_path: ~/.cache/models/vision/yolov8/yolov8_fire.q.onnx"
