# 跌倒检测应用

单目标：姿态检测后只跟踪当前帧 **score 最高** 的一人，用 STGCN（30 帧关键点序列）识别动作/跌倒。

## 配置

- **入口**：`config/fall_detection.yaml`（模型引用 + `test_video`、`kp_threshold`、`stgcn_*`）
- **模型**：`pose.yaml`、`stgcn.yaml`
  - 姿态：`yolov8n-pose.q.onnx`（SpaceMIT EP）
  - 动作（默认）：`stgcn.dynq.onnx` + SpaceMIT EP（见 `config/stgcn.yaml`）
  - 动作（备选）：`stgcn.fp32.onnx` + `CPUExecutionProvider`（修改 `config/stgcn.yaml` 中 `model_path` 与 `providers`）
- **`fall_down_index`**（应用策略）：STGCN 输出的哪一类算「跌倒」、触发告警。这是本应用的业务判定，不属于模型，故放在 `fall_detection.yaml`。STGCN 类名顺序见 `assets/labels/stgcn.txt`（默认 `Fall Down` 在下标 6）。

## 下载模型

```bash
sh applications/fall_detection/scripts/download_models.sh
```

会下载姿态估计与 STGCN 两个版本（`stgcn.dynq.onnx`、`stgcn.fp32.onnx`）到 `~/.cache/models/vision/`，可按需切换 `config/stgcn.yaml`。

## 运行

```bash
python applications/fall_detection/python/example_fall_detection.py
./applications/example_fall_detection
```

CLI：`--config`、`--video`、`--use-camera`、`--camera-id`。C++ 另支持位置参数 `[config.yaml]`。
