# MobileNetV1 示例

ImageNet 图像分类示例，使用 `deploy.mobilenet.MobileNetV1Classifier`。

MobileNetV1 (TF-Slim 导出) 输出 1001 类，其中 index 0 为 background。本示例的 deploy 实现会截掉背景类，使剩余 1000 个 logits 与标准 ImageNet 标签对齐。

> MobileNetV2 / V3 预处理标准、输出 1000 类，见 `examples/mobilenet`。

## 1. 模型与权重

- **默认模型**：`~/.cache/models/vision/mobilenet/mobilenet_v1.q.onnx`
- **下载**（在仓库根目录执行）：`bash examples/mobilenetv1/scripts/download_models.sh`

**测试图片**：`~/.cache/assets/image/005_kitten.jpg`。若未下载资源，在 cv 根目录执行 `bash scripts/download_assets.sh`。

## 2. 运行示例

**Python：**

```bash
python examples/mobilenetv1/python/mobilenetv1.py --config examples/mobilenetv1/config/mobilenetv1.yaml
```

**C++：** 在 `build` 目录下：

```bash
./examples/mobilenetv1 examples/mobilenetv1/config/mobilenetv1.yaml
```

## 3. 故障排查

- **模型未找到**：确认已执行 `bash examples/mobilenetv1/scripts/download_models.sh`。
- **标签错位**：确认使用 1000 类 `assets/labels/imagenet.txt`，背景类已由 deploy 实现截除。
