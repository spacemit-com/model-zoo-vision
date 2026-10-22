# EfficientNet-V2-S 示例

EfficientNet-V2-S 的 ImageNet 图像分类示例。输出类别及分数，示例在终端打印分类结果。

## 1. 模型与权重

- **模型类型**：图像分类
- **默认模型文件**：`~/.cache/models/vision/efficientnet/efficientnet_v2_s.q.onnx`，与 `config/efficientnet_v2s.yaml` 一致。
- **下载**：在仓库根目录执行：

```bash
bash examples/efficientnet_v2s/scripts/download_models.sh
```

该配置将图像直接缩放到 224×224，不做中心裁剪。B0/B1 使用不同的预处理，应使用 [EfficientNet 示例](../efficientnet/README.md) 的配置。

**测试图片**：默认使用 `~/.cache/assets/image/005_kitten.jpg`。资源未下载时，在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/efficientnet_v2s.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/efficientnet/efficientnet_v2_s.q.onnx` |
| `test_image` | 默认测试图片 | `~/.cache/assets/image/005_kitten.jpg` |
| `label_file_path` | ImageNet 类别标签 | `assets/labels/imagenet.txt` |
| `default_params.num_threads` | 推理线程数 | `8` |
| `default_params.resize_size` | 裁剪或缩放前的尺寸，顺序为 [width, height] | `[224, 224]` |
| `default_params.mean` | RGB 通道归一化均值 | `[0.485, 0.456, 0.406]` |
| `default_params.std` | RGB 通道归一化标准差 | `[0.229, 0.224, 0.225]` |
| `default_params.center_crop` | 是否执行中心裁剪 | `false` |
| `default_params.interpolation` | 缩放插值方式 | `bicubic` |
| `default_params.providers` | 执行提供方，单元素列表 | `[SpaceMITExecutionProvider]` |

输入尺寸由 ONNX 模型决定。`resize_size` 指预处理尺寸，不能用来修改模型的输入形状；
该字段必须是两个正整数。`interpolation` 支持 `bilinear/linear`、`bicubic/cubic`、
`nearest`、`area`。

`default_params.apply_softmax` 为 `true`（未配置时的代码默认值），用于匹配权重的输出：logits 使用
`true`，已经包含 Softmax 的概率输出使用 `false`。该设置在创建模型时生效。
更换权重时，应同时核对裁剪、均值、标准差和输出类别数。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/efficientnet_v2s.py`）：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/efficientnet_v2s/config/efficientnet_v2s.yaml` |
| `--image` | 输入图片路径 | YAML 中的 `test_image` |
| `--top-k` | 在终端显示前 K 个类别 | `5` |
| `--model-path` | 覆盖主模型路径 | 无 |

**C++ 示例**：第一个参数是配置文件路径，支持 `--image`、`--model-path`，
默认打印前 5 个类别；当前没有 `--top-k` 命令行选项。

通过 API 控制本次推理返回的类别数：

```python
from spacemit_vision import VisionServiceInferParams

params = VisionServiceInferParams()
params.top_k = 3
status, results = service.infer_image(image, params=params)
```

`top_k` 负数使用模型默认值 5，0 返回空结果，正数限制返回类别数。
它是运行时参数，不要写入 YAML 的 `default_params`。
Python 示例的 `--top-k` 对完整类别分数排序后控制显示数量，与 API 的返回数量设置是两个入口。

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

**Python：** 在仓库根目录运行：

```bash
python examples/efficientnet_v2s/python/efficientnet_v2s.py --config examples/efficientnet_v2s/config/efficientnet_v2s.yaml
python examples/efficientnet_v2s/python/efficientnet_v2s.py --config examples/efficientnet_v2s/config/efficientnet_v2s.yaml --image /path/to/image.jpg --top-k 3
```

**C++：** 在 `build` 目录运行：

```bash
./examples/efficientnet_v2s ../examples/efficientnet_v2s/config/efficientnet_v2s.yaml
./examples/efficientnet_v2s ../examples/efficientnet_v2s/config/efficientnet_v2s.yaml --image /path/to/image.jpg
```

## 5. 故障排查

- **模型未找到**：执行 `bash examples/efficientnet_v2s/scripts/download_models.sh`，或用 `--model-path` 指定兼容的权重。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或用 `--image` 指定图片。
- **标签文件未找到**：检查 `label_file_path` 指向的 `assets/labels/imagenet.txt`。
- **分类结果异常**：核对权重对应的归一化、裁剪、类别数和 `apply_softmax` 设置。
- **Python 扩展导入失败**：安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。
