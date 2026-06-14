# 跌倒检测应用

## 配置

- **入口**：`config/fall_detection.yaml`（模型引用 + `test_video`、`kp_threshold`、`stgcn_*`）
- **模型**：`pose.yaml`、`stgcn.yaml`
- **`fall_down_index`**（应用策略）：STGCN 输出的哪一类算「跌倒」、触发告警。这是本应用的业务判定，不属于模型，故放在 `fall_detection.yaml`。STGCN 类名顺序见 `assets/labels/stgcn.txt`（默认 `Fall Down` 在下标 6）。

## 运行

```bash
python applications/fall_detection/python/example_fall_detection.py
./applications/example_fall_detection
```

CLI：`--config`、`--video`、`--use-camera`、`--camera-id`。C++ 另支持位置参数 `[config.yaml]`。
