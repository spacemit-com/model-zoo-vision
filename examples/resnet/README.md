## ResNet 示例

ImageNet 图像分类。默认 `resnet50.q.onnx`；下载脚本同时提供 `resnet18.q.onnx`、`resnet50.fp16.onnx`、`resnet50.b4.q.onnx`（batch=4，需自行组 batch 推理）。

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
./examples/resnet examples/resnet/config/resnet50.yaml
```

