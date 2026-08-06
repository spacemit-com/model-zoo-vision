# MobileSeg 示例

语义分割示例：使用 MobileSeg MobileNetV2 对图像执行 Cityscapes
19 类逐像素分类，并保存叠加分割掩码后的结果图。

## 1. 模型与权重

- **模型类型**：语义分割（MobileSeg，MobileNetV2 骨干）
- **训练数据集**：Cityscapes，19 类
- **固定输入**：RGB `float32 [1, 3, 512, 1024]`
- **模型输出**：类别图 `int32 [1, 512, 1024]`
- **默认模型文件**：`~/.cache/models/vision/mobileseg/mobileseg_mobilenetv2_cityscapes_1024x512.q.onnx`

在组件根目录执行：

```bash
bash examples/mobileseg/scripts/download_models.sh
```

脚本会校验模型 SHA-256，并优先复制组件根目录下本次导入提供的同名
模型；若本地模型不存在，则从 SpaceMIT 模型归档下载。也可分别使用
`MOBILESEG_MODEL_SOURCE` 或 `MOBILESEG_BASE_URL` 指定本地模型和下载
地址。

默认测试图片为 `~/.cache/assets/image/009_test_unet.jpg`。若图片尚未
准备，在组件根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/mobileseg.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/mobileseg/mobileseg_mobilenetv2_cityscapes_1024x512.q.onnx` |
| `test_image` | 默认测试图片 | `~/.cache/assets/image/009_test_unet.jpg` |
| `label_file_path` | Cityscapes 19 类标签 | `assets/labels/cityscapes.txt` |
| `class` | 工厂注册类 | `deploy.mobileseg.MobileSeg` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `8` |
| `default_params.num_classes` | 输出类别数 | `19` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |
| `default_params.preprocess.backend` | 图像预处理后端 | `cpu` |

预处理与 PaddleSeg 导出契约一致：输入图像直接拉伸至
`1024x512`，BGR 转 RGB，并执行 `(pixel / 255 - 0.5) / 0.5`。
模型已经内置 `ArgMax`，输出是类别编号而不是 logits。类别图用最近邻
插值还原至原图大小，类别 `0`（road）也会作为有效掩码返回。

## 3. 命令行 / API 参数（与本模块相关）

Python 示例支持：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/mobileseg/config/mobileseg.yaml` |
| `--image` | 覆盖输入图片 | yaml 中的 `test_image` |
| `--output` | 输出可视化图片 | `mobileseg_result.jpg` |
| `--model-path` | 覆盖 yaml 中的模型路径 | 无 |

C++ 示例的第一个参数必须是配置文件路径，并支持相同的 `--image`、
`--output` 和 `--model-path` 覆盖参数。

模型通过统一的 `VisionService` 创建，支持 `ImageInput`、分割意图和
`Draw()`。返回结果是当前图像实际出现的类别掩码，每个掩码尺寸与原图
一致。

## 4. 运行示例

在组件根目录配置并编译 C++ example 和 Python 原生扩展：

```bash
cmake -S . -B build
cmake --build build --target mobileseg --target _vision_service_cpp -j
```

运行默认图片：

```bash
./build/examples/mobileseg \
  examples/mobileseg/config/mobileseg.yaml
```

Python 示例使用同一个 C++ `VisionService`。使用已安装的
`spacemit_vision` wheel 时执行：

```bash
python3 examples/mobileseg/python/mobileseg.py
```

直接使用当前源码树和 CMake 构建产物时执行：

```bash
PYTHONPATH=build/python:src/python \
  python3 examples/mobileseg/python/mobileseg.py
```

指定输入和输出：

```bash
./build/examples/mobileseg \
  examples/mobileseg/config/mobileseg.yaml \
  --image /path/to/image.jpg \
  --output mobileseg_result.jpg

PYTHONPATH=build/python:src/python \
  python3 examples/mobileseg/python/mobileseg.py \
  --image /path/to/image.jpg \
  --output mobileseg_result.jpg
```

## 5. 故障排查

- **模型未找到**：执行 `bash examples/mobileseg/scripts/download_models.sh`，
  或用 `--model-path` 指定同一导出契约的模型。
- **模型校验失败**：确认使用的是本示例对应的量化 ONNX；不要用其他
  MobileSeg 权重覆盖同名文件。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或传入
  `--image`。
- **模型形状或类型错误**：本实现只接受 `float32 [1,3,H,W]` 输入和
  空间尺寸一致的 `int32 [1,H,W]` 类别图输出。
- **颜色或分割结果异常**：确认输入是 OpenCV BGR 三通道图片；预处理
  会在内部完成 RGB 转换和 `0.5/0.5` 归一化。
- **Python 无法导入原生扩展**：先构建 `_vision_service_cpp`，并使用
  已安装的 wheel，或按上面的 `PYTHONPATH=build/python:src/python`
  命令从源码树运行。
