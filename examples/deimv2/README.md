# DEIMv2-N 示例

COCO 目标检测示例：对图像执行 DEIMv2-N 推理，输出检测框、置信度与类别标签，
并保存可视化结果。

## 1. 模型与权重

- **模型类型**：目标检测（DEIMv2-N）
- **默认模型文件**：`~/.cache/models/vision/deimv2/deimv2n.fp16.onnx`
- **可选模型文件**：`deimv2s.fp16.onnx`、`deimv2m.fp16.onnx`
- **默认测试图片**：`~/.cache/assets/image/006_test.jpg`
- **类别标签文件**：`assets/labels/coco.txt`

在组件根目录执行：

```bash
bash examples/deimv2/scripts/download_models.sh
bash scripts/download_assets.sh
```

下载脚本会准备 N、S、M 三套模型。本地存在同名导入文件时优先复制，否则从
归档服务器的 `vision/deimv2/` 下载；所有文件都会校验 SHA-256。

### 模型输入输出契约

模型包含两个输入和三个已解码输出：

| 名称 | 类型与形状 | 说明 |
|------|------------|------|
| `images` | float32 `[1, 3, 640, 640]` | RGB；N 做 `/255`，S/M 再做 ImageNet mean/std |
| `orig_target_sizes` | int64 `[1, 2]` | 传入预处理画布尺寸 `[640, 640]` |
| `labels` | int64 `[1, 300]` | COCO 类别索引 |
| `boxes` | float32 `[1, 300, 4]` | 640×640 画布上的 xyxy 框 |
| `scores` | float16 `[1, 300]` | 检测置信度 |

预处理采用保持宽高比的居中 letterbox，填充值为 0。后处理移除 padding 并将框
还原到原图坐标。模型图内已经完成 TopK 和框解码，因此示例只按置信度过滤，
不额外执行 NMS。统一检测接口中的 `iou_threshold` 仅为兼容接口签名保留，
DEIMv2 推理不会使用该参数。

## 2. 配置文件说明（config/deimv2.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/deimv2/deimv2n.fp16.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/006_test.jpg` |
| `label_file_path` | COCO 类别标签文件 | `assets/labels/coco.txt` |
| `image_size` | 模型输入尺寸 | `[640, 640]` |
| `default_params.conf_threshold` | 置信度阈值 | `0.4` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `8` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |
| `default_params.preprocess.backend` | 预处理策略：`cpu` / `auto` / `opencl` | `auto` |
| `default_params.preprocess.normalize` | 按模型文件名识别归一化，也可显式设为 `true` / `false` | `auto` |

说明：通用字段（如 `class`、`default_params` 结构）与其他示例一致，
不在此重复。

默认配置使用 N。切换到 S/M 时通过 `--model-path` 指定模型即可，检测器会按
`deimv2n`、`deimv2s`、`deimv2m` 文件名自动选择归一化方式。若模型被重命名，
需在配置中将 `normalize` 显式设为 `true` 或 `false`。

## 3. 命令行参数（与本模块相关）

Python 示例支持：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/deimv2/config/deimv2.yaml` |
| `--image` | 覆盖输入图片 | yaml 中的 `test_image` |
| `--output` | 输出可视化图片 | `deimv2_result.jpg` |
| `--model-path` | 覆盖 yaml 中的模型路径 | 无 |
| `--conf-threshold` | 覆盖置信度阈值 | 使用 yaml 配置 |

C++ 示例的第一个参数必须是配置文件路径，并支持 `--image`、`--output`
和 `--model-path`。

## 4. 运行示例

在组件根目录运行 C++ 示例：

```bash
./build/examples/deimv2 examples/deimv2/config/deimv2.yaml
./build/examples/deimv2 examples/deimv2/config/deimv2.yaml \
  --model-path ~/.cache/models/vision/deimv2/deimv2s.fp16.onnx
./build/examples/deimv2 examples/deimv2/config/deimv2.yaml \
  --model-path ~/.cache/models/vision/deimv2/deimv2m.fp16.onnx
./build/examples/deimv2 examples/deimv2/config/deimv2.yaml \
  --image /path/to/image.jpg --output deimv2_result.jpg
```

使用已安装的 `spacemit_vision` Python 包。运行前需确认当前 Python 环境已经
安装本仓库构建出的 wheel：

```bash
python3 examples/deimv2/python/deimv2.py
python3 examples/deimv2/python/deimv2.py \
  --model-path ~/.cache/models/vision/deimv2/deimv2s.fp16.onnx
```

直接使用当前源码树。运行前需先构建原生扩展，并通过 `build_wheel.sh` 将最新的
扩展和 `libvision.so` 更新到源码包目录：

```bash
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build -j8 --target _vision_service_cpp
(cd src/python && PYTHON_BIN=python3 ./build_wheel.sh ../../build)
PYTHONPATH=src/python \
  python3 examples/deimv2/python/deimv2.py
```

## 5. 故障排查

- **模型未找到**：执行 `bash examples/deimv2/scripts/download_models.sh`，
  或使用 `--model-path` 指定模型。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或使用
  `--image` 指定图片。
- **模型创建失败**：确认模型输入输出名称和形状与上表一致，并确认配置中的
  执行提供方可用。
- **Python 无法导入原生扩展**：先构建 `_vision_service_cpp`，并使用已安装的
  Python 包，或按上面的 `PYTHONPATH` 命令从源码树运行。
