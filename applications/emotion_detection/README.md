# 情绪检测应用

支持两种模式：

- **image 模式（静态）**：YOLOv5-Face 检测 + ResNet50 单帧情绪分类
- **camera 模式（动态）**：YOLOv5-Face 检测 + ResNet50 特征提取 + LSTM 时序情绪识别（10 帧滑窗）

动态模式需要连续帧累积，单帧没有意义，因此只在摄像头实时场景启用。

## 配置

- **入口**：`config/emotion_detection.yaml`
  - `face_model: face_detector.yaml` — 人脸检测（两模式共用）
  - `emotion_model: emotion.yaml` — 静态情绪 ResNet50（image 模式）
  - `feature_model: emotion_features.yaml` — ResNet50 特征提取 backbone（camera 模式，`EmotionRecognizer` 特征模式）
  - `lstm_model: emotion_lstm.yaml` — LSTM 时序分类（camera 模式）
- **模型 schema**：同目录下各子 yaml

camera 模式的滑窗（10 帧 512 维特征）在应用层维护：backbone 逐帧提特征，攒满 10 帧后交给 LSTM 出 7 类概率。
两个动态模型（features + LSTM）使用相同的 EP 与线程数。

## 下载模型

```bash
sh applications/emotion_detection/scripts/download_models.sh
```

会下载 4 个模型：人脸检测、静态 ResNet50、ResNet50 特征提取器、LSTM 分类器。

## 运行

```bash
# image 模式（静态，单帧）
python applications/emotion_detection/python/example_emotion.py --image /path/to.jpg

# camera 模式（动态 LSTM，实时）
python applications/emotion_detection/python/example_emotion.py --use-camera
python applications/emotion_detection/python/example_emotion.py --use-camera --camera-id 1
```

camera 模式下，前 10 帧累积特征显示 `buffering N/10`，之后输出滑窗情绪标签。按 `q` 退出，`s` 保存当前帧。

C++（需在带 SpaceMIT 工具链的设备上 `cmake -B build && cmake --build build`）：

```bash
./build/applications/example_emotion --image /path/to.jpg      # image 模式
./build/applications/example_emotion --use-camera              # camera 模式
```

静态与动态模式共用 `assets/labels/emotion.txt`（Emo-AffectNet 类别顺序）。

CLI：`--config`、`--image`、`--output`、`--use-camera`、`--camera-id`。
