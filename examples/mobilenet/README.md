# MobileNet 示例

ImageNet 图像分类示例，复用 `deploy.resnet.ResNetClassifier`。

## 1. 模型与权重

- **默认模型**：`~/.cache/models/vision/mobilenet/mobilenet_v2.q.onnx`
- **下载**（在仓库根目录执行）：`bash examples/mobilenet/scripts/download_models.sh`

可选规格：`mobilenet_v1.q.onnx`、`mobilenet_v2.q.onnx`、`mobilenet_v3_small.fp16.onnx`、`mobilenet_v3_large.fp16.onnx`。

> 注意：`mobilenet_v1.q.onnx` 输出为 1001 类（含 background），与其余 1000 类 ImageNet 标签不完全对齐。

**测试图片**：`~/.cache/assets/image/005_kitten.jpg`。若未下载资源，在 cv 根目录执行 `bash scripts/download_assets.sh`。

## 2. 运行示例

**Python：**

```bash
python examples/mobilenet/python/mobilenet.py --config examples/mobilenet/config/mobilenet.yaml
```

**C++：** 在 `build` 目录下：

```bash
./examples/mobilenet examples/mobilenet/config/mobilenet.yaml
```

## 3. 故障排查

- **模型未找到**：确认已执行 `bash examples/mobilenet/scripts/download_models.sh`。
