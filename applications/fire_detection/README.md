# 火焰检测应用

## 配置

- **入口**：`config/fire_detection.yaml`（`model: yolov8_fire.yaml`）
- **模型**：`config/yolov8_fire.yaml`

## 运行

```bash
python applications/fire_detection/python/example_fire_detection.py
./applications/example_fire_detection
```

CLI：`--config`、`--image` / `--video` / `--use-camera`、`--output`（Python）。C++ 另支持位置参数 `[config.yaml]`。
