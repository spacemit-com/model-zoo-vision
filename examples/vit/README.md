# ViT 示例

Vision Transformer（ViT-B/16）ImageNet 图像分类示例，复用 `deploy.resnet.ResNetClassifier`。

## 1. 模型与权重

- **默认模型**：`~/.cache/models/vision/vit/vit_b_16.q.onnx`
- **下载**（在仓库根目录执行）：`bash examples/vit/scripts/download_models.sh`

可选规格：`vit_b_16.q.onnx`、`vit_b_16.fp16.onnx`。

**测试图片**：`~/.cache/assets/image/005_kitten.jpg`。若未下载资源，在 cv 根目录执行 `bash scripts/download_assets.sh`。

## 2. 运行示例

**Python：**

```bash
python examples/vit/python/vit.py --config examples/vit/config/vit.yaml
```

**C++：** 在 `build` 目录下：

```bash
./examples/vit examples/vit/config/vit.yaml
```

## 3. 故障排查

- **模型未找到**：确认已执行 `bash examples/vit/scripts/download_models.sh`。
