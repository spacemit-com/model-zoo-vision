# YOLO26 示例

目标检测示例（YOLO26 系列），使用独立的 `deploy.yolo26.YOLO26Detector`。

仅支持 e2e 导出格式：

- **e2e**：`[N, 6]` 或 `[1, N, 6]`（`x1,y1,x2,y2,score,class`，模型已带 NMS）

## 1. 模型与权重

- 默认模型：`~/.cache/models/vision/yolo26/yolo26n.q.onnx`
- 下载（仓库根目录）：`bash examples/yolo26/scripts/download_models.sh`

## 2. 运行示例

Python:

```bash
python examples/yolo26/python/yolo26.py --config examples/yolo26/config/yolo26.yaml
python examples/yolo26/python/yolo26.py --config examples/yolo26/config/yolo26.yaml --image /path/to/image.jpg --output result.jpg
```

C++（在 `build` 目录）:

```bash
./examples/yolo26 examples/yolo26/config/yolo26.yaml
./examples/yolo26 examples/yolo26/config/yolo26.yaml --image /path/to/image.jpg --output result.jpg
```

## 3. 故障排查

- 模型未找到：先执行 `bash examples/yolo26/scripts/download_models.sh`
- 默认测试图缺失：在仓库根目录执行 `bash scripts/download_assets.sh`
