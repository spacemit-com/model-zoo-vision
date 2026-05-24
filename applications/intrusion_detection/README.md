# 区域闯入识别应用

## 配置

- **入口**：`config/intrusion_detection.yaml`（`tracker_model` + `test_video`）
- **模型**：`bytetrack.yaml`

## 运行

```bash
python applications/intrusion_detection/python/example_intrusion_identification.py
./applications/example_intrusion_identification
```

CLI：`--config`、`--video`、`--use-camera`；Python 另支持 `--roi-points`、`--counting-mode`。C++ 另支持位置参数 `[config.yaml]`。
