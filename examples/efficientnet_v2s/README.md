# EfficientNet-V2-S 示例

ImageNet 图像分类示例，使用 `deploy.resnet.ResNetClassifier`，但预处理参数不同。

EfficientNet-V2-S 直接 resize 到 224、**不做 center crop**（`center_crop: false`、`resize_size: [224, 224]`），与 B0/B1 的「resize 256 + crop 224」不同，因此独立成 example。

> EfficientNet-B0 / B1 见 `examples/efficientnet`。

## 1. 模型与权重

- **默认模型**：`~/.cache/models/vision/efficientnet/efficientnet_v2_s.q.onnx`
- **下载**（在仓库根目录执行）：`bash examples/efficientnet_v2s/scripts/download_models.sh`

**测试图片**：`~/.cache/assets/image/005_kitten.jpg`。若未下载资源，在 cv 根目录执行 `bash scripts/download_assets.sh`。

## 2. 运行示例

**Python：**

```bash
python examples/efficientnet_v2s/python/efficientnet_v2s.py --config examples/efficientnet_v2s/config/efficientnet_v2s.yaml
```

**C++：** 在 `build` 目录下：

```bash
./examples/efficientnet_v2s examples/efficientnet_v2s/config/efficientnet_v2s.yaml
```

## 3. 备注

- 若 top-1 结果异常，尝试将 `mean`/`std` 改为 `[0.5, 0.5, 0.5]`（部分 V2-S 权重使用 [-1,1] 归一化）。
