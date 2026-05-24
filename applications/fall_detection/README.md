# 跌倒检测应用

## 配置

- **入口**：`config/fall_detection.yaml`（模型引用 + `test_video`、`kp_threshold`、`stgcn_*`）
- **模型**：`pose.yaml`、`stgcn.yaml`

## 运行

```bash
python applications/fall_detection/python/example_fall_detection.py
./applications/example_fall_detection
```

CLI：`--config`、`--video`、`--use-camera`、`--camera-id`。C++ 另支持位置参数 `[config.yaml]`。
