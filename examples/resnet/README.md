## ResNet 示例

ImageNet 图像分类。默认 `resnet50.q.onnx`；下载脚本同时提供 `resnet18.q.onnx`、`resnet50.fp16.onnx`、`resnet50.b4.q.onnx`（batch=4，需自行组 batch 推理）。

`default_params.apply_softmax` 默认为 `true`，适用于输出 logits 的权重；
若替换的权重已输出概率，应设为 `false`。复用本实现的 EfficientNet、ViT、MobileNet V2/V3
使用相同配置语义。每次推理的 `top_k` 可指定返回类别数；负数使用默认 5，0 返回空结果。

`top_k` 通过 `VisionServiceInferParams.top_k` 传给 `service.infer(image, params)`，
也可传给 `service.infer_image(image, params=params)`，不要写入 `default_params`。
`apply_softmax` 在 YAML 中设置，创建后保持固定。
`resize_size` 使用 `[width, height]`，必须包含两个正整数；`interpolation`
支持 `bilinear/linear`、`bicubic/cubic`、`nearest`、`area`，拼写错误会报错。

### 下载模型

在本目录执行：

```bash
bash scripts/download_models.sh
```

模型保存到 `~/.cache/models/vision/resnet/`。

### 运行示例

- Python：

```bash
python examples/resnet/python/resnet.py --config examples/resnet/config/resnet50.yaml
```

- C++（在 build 目录）：

```bash
./examples/resnet ../examples/resnet/config/resnet50.yaml
```

