# YOLOP 示例

道路场景多任务示例：对图像同时执行车辆检测、可行驶区域分割和车道线分割，
并保存可视化结果。

## 1. 模型与权重

- **模型类型**：道路场景多任务感知（YOLOP）
- **默认模型文件**：`~/.cache/models/vision/yolop/yolop-288-512-sim.q.onnx`
- **默认测试图片**：`~/.cache/assets/image/020_yolop.jpg`
- **类别标签文件**：`assets/labels/yolop.txt`

在组件根目录执行：

```bash
bash examples/yolop/scripts/download_models.sh
bash scripts/download_assets.sh
```

模型下载脚本会将模型保存到上述缓存路径。默认测试图片也会由下载脚本准备，
已有文件不会重复下载。

## 2. 配置文件说明（config/yolop.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/yolop/yolop-288-512-sim.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/020_yolop.jpg` |
| `label_file_path` | 类别标签文件 | `assets/labels/yolop.txt` |
| `image_size` | 模型输入尺寸 | `[288, 512]` |
| `default_params.conf_threshold` | 置信度阈值 | `0.25` |
| `default_params.iou_threshold` | NMS IoU 阈值 | `0.45` |
| `default_params.max_det` | 最大检测数量 | `300` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `8` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

说明：通用字段（如 `class`、`default_params` 结构）与其他示例一致，
不在此重复。

## 3. 命令行参数（与本模块相关）

Python 示例支持：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/yolop/config/yolop.yaml` |
| `--image` | 覆盖输入图片 | yaml 中的 `test_image` |
| `--output` | 输出可视化图片 | `yolop_result.jpg` |
| `--model-path` | 覆盖 yaml 中的模型路径 | 无 |
| `--conf-threshold` | 覆盖置信度阈值 | 使用 yaml 配置 |
| `--iou-threshold` | 覆盖 NMS IoU 阈值 | 使用 yaml 配置 |

C++ 示例的第一个参数必须是配置文件路径，并支持 `--image`、`--output`
和 `--model-path`。

## 4. 运行示例

在组件根目录运行 C++ 示例：

```bash
./build/examples/yolop examples/yolop/config/yolop.yaml
./build/examples/yolop examples/yolop/config/yolop.yaml \
  --image /path/to/image.jpg --output yolop_result.jpg
```

使用已安装的 `spacemit_vision` Python 包：

```bash
python3 examples/yolop/python/yolop.py
```

直接使用当前源码树和 CMake 构建产物：

```bash
PYTHONPATH=build/python:src/python \
  python3 examples/yolop/python/yolop.py

PYTHONPATH=build/python:src/python \
  python3 examples/yolop/python/yolop.py \
  --image /path/to/image.jpg --output yolop_result.jpg
```

## 5. 故障排查

- **模型未找到**：执行 `bash examples/yolop/scripts/download_models.sh`，
  或使用 `--model-path` 指定模型。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或使用
  `--image` 指定图片。
- **模型创建失败**：确认配置中的模型路径、模型输入尺寸和
  `SpaceMITExecutionProvider` 可用。
- **Python 无法导入原生扩展**：先构建 `_vision_service_cpp`，并使用已安装的
  Python 包，或按上面的 `PYTHONPATH` 命令从源码树运行。
