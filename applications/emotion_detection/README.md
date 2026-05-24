# 情绪检测应用

## 配置

- **入口**：`config/emotion_detection.yaml`（引用 `face_detector.yaml`、`emotion.yaml`）
- **模型 schema**：同目录下 `face_detector.yaml`、`emotion.yaml`

## 运行

```bash
# 默认配置即可
python applications/emotion_detection/python/example_emotion.py
./applications/example_emotion

# 可选覆盖
python applications/emotion_detection/python/example_emotion.py --config path/to/emotion_detection.yaml --image /path/to.jpg
./applications/example_emotion --image /path/to.jpg [output.jpg]
```

CLI：`--config`、`--image`、`--use-camera`、`--camera-id`（及 C++ 位置参数 output）。
