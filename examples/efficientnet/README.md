# EfficientNet 示例

ImageNet 图像分类示例，复用 `deploy.resnet.ResNetClassifier`（ImageNet 预处理与 ResNet 一致）。

## 1. 模型与权重

- **默认模型**：`~/.cache/models/vision/efficientnet/efficientnet_v1_b0.q.onnx`
- **下载**（在仓库根目录执行）：`bash examples/efficientnet/scripts/download_models.sh`

可选规格：`efficientnet_v1_b0/b1`（`.q.onnx` / `.fp16.onnx`）、`efficientnet_v2_s`（`.q.onnx` / `.fp16.onnx`）。通过 `--model-path` 或修改 `config/efficientnet.yaml` 中的 `model_path` 切换。

**测试图片**：`~/.cache/assets/image/005_kitten.jpg`。若未下载资源，在 cv 根目录执行 `bash scripts/download_assets.sh`。

## 2. 运行示例

**Python：**

```bash
python examples/efficientnet/python/efficientnet.py --config examples/efficientnet/config/efficientnet.yaml
```

**C++：** 在 `build` 目录下：

```bash
./examples/efficientnet examples/efficientnet/config/efficientnet.yaml
```

## 3. 故障排查

- **模型未找到**：确认已执行 `bash examples/efficientnet/scripts/download_models.sh`。
- **标签文件未找到**：确认 `assets/labels/imagenet.txt` 存在。
