#!/bin/sh
# 跌倒检测应用：下载模型到 ~/.cache/models/vision/
# 运行: sh download_models.sh（在 scripts 目录或应用根目录下执行）

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
MODEL_DIR="$CACHE_BASE"
mkdir -p "$MODEL_DIR/yolov8_pose" "$MODEL_DIR/stgcn"
MODEL_DIR_POSE="$CACHE_BASE/yolov8_pose"
MODEL_DIR_STGCN="$CACHE_BASE/stgcn"

download_base() {
  base_url="$1"
  dir="$2"
  name="$3"
  if [ -f "$dir/$name" ]; then
    echo "已存在: $dir/$name"
    return 0
  fi
  echo "下载 $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$dir/$name" "$base_url/$name"
  else
    wget -O "$dir/$name" "$base_url/$name"
  fi
}

BASE_POSE="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov8_pose"
download_base "$BASE_POSE" "$MODEL_DIR_POSE" "yolov8n-pose.q.onnx"

BASE_STGCN="https://archive.spacemit.com/spacemit-ai/model_zoo/vision/stgcn"
download_base "$BASE_STGCN" "$MODEL_DIR_STGCN" "stgcn.onnx"

echo "完成。模型目录: $CACHE_BASE"
echo "配置中请使用:"
echo "  pose_model_path: ~/.cache/models/vision/yolov8_pose/yolov8n-pose.q.onnx"
echo "  stgcn_model_path: ~/.cache/models/vision/stgcn/stgcn.onnx"
