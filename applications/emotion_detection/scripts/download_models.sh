#!/bin/sh
# 情绪检测应用：下载模型到 ~/.cache/models/vision/
# 运行: sh download_models.sh（在 scripts 目录或应用根目录下执行）

set -e
CACHE_BASE="${HOME:-/tmp}/.cache/models/vision"
mkdir -p "$CACHE_BASE/yolov5-face" "$CACHE_BASE/emotion"

download() {
  url="$1"
  dir="$2"
  name="$3"
  if [ -f "$dir/$name" ]; then
    echo "已存在: $dir/$name"
    return 0
  fi
  echo "下载 $name ..."
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$dir/$name" "$url"
  else
    wget -O "$dir/$name" "$url"
  fi
}

download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/emotion/emotion_resnet50_final.q.onnx" "$CACHE_BASE/emotion" "emotion_resnet50_final.q.onnx"
download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolov5-face/yolov5n-face_cut.q.onnx" "$CACHE_BASE/yolov5-face" "yolov5n-face_cut.q.onnx"
# camera 模式（动态 LSTM）所需：ResNet50 特征提取 + LSTM 时序分类
download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/emotion/emotion_resnet50_features.dynq.onnx" "$CACHE_BASE/emotion" "emotion_resnet50_features.dynq.onnx"
download "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/emotion/emotion_lstm_affwild2_final.q.onnx" "$CACHE_BASE/emotion" "emotion_lstm_affwild2_final.q.onnx"

echo "完成。模型目录: $CACHE_BASE"
echo "情绪模型统一在: ~/.cache/models/vision/emotion/"
echo "  emotion_resnet50_final.q.onnx"
echo "  emotion_resnet50_features.dynq.onnx"
echo "  emotion_lstm_affwild2_final.q.onnx"
echo "人脸检测: ~/.cache/models/vision/yolov5-face/yolov5n-face_cut.q.onnx"
