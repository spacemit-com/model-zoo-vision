# YOLO12 示例

目标检测示例（YOLO12 系列）。模型为单输出格式 `[1, 84, 8400]`，与 YOLOv8 官方导出一致，复用 `deploy.yolov8.YOLOv8Detector`。

## 1. 模型与权重

- **默认模型**：`~/.cache/models/vision/yolo12/yolo12n.q.onnx`
- **下载**（在仓库根目录执行）：`bash examples/yolo12/scripts/download_models.sh`

可选规格：`yolo12n.q.onnx`、`yolo12s.q.onnx`、`yolo12m.q.onnx`。

**测试图片**：`~/.cache/assets/image/006_test.jpg`。若未下载资源，在 cv 根目录执行 `bash scripts/download_assets.sh`。

## 2. 运行示例

**Python：**

```bash
python examples/yolo12/python/yolo12.py --config examples/yolo12/config/yolo12.yaml
python examples/yolo12/python/yolo12.py --config examples/yolo12/config/yolo12.yaml --image /path/to/image.jpg --output result.jpg
```

**C++：** 在 `build` 目录下：

```bash
./examples/yolo12 examples/yolo12/config/yolo12.yaml
./examples/yolo12 examples/yolo12/config/yolo12.yaml --image test.jpg --output result.jpg
```

## 3. 故障排查

- **模型未找到**：确认已执行 `bash examples/yolo12/scripts/download_models.sh`，且 `model_path` 指向正确文件。
