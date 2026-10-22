# YOLOP 示例

道路场景多任务示例：对图像同时执行车辆检测、可行驶区域分割和车道线分割，
并保存可视化结果。

## 1. 模型与权重

- **模型类型**：道路场景多任务感知（YOLOP）
- **默认模型文件**：`~/.cache/models/vision/yolop/yolop-288-512-sim.q.onnx`
- **默认测试图片**：`~/.cache/assets/image/020_yolop.jpg`
- **类别标签文件**：`assets/labels/yolop.txt`

在仓库根目录执行：

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
| ONNX 输入尺寸 | 由模型自动读取，不能通过 YAML 改写 | 见模型文件 |
| `default_params.conf_threshold` | 置信度阈值 | `0.25` |
| `default_params.iou_threshold` | NMS IoU 阈值 | `0.45` |
| `default_params.max_det` | 最大检测数量 | `300` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `8` |
| `default_params.providers` | ONNX Runtime 执行提供方（单元素列表） | `[SpaceMITExecutionProvider]` |

说明：通用字段（如 `class`、`default_params` 结构）与其他示例一致，
不在此重复。

`default_params.max_det` 设置默认检测数量上限，建议使用正整数。
单次请求可通过 `VisionServiceInferParams.max_det` 覆盖：负数使用 YAML 默认值，
0 不返回检测框（仍保留分割输出），正数设置本次数量上限。
`conf_threshold` 的正数请求值覆盖
YAML 默认值，`<=0` 使用默认值。

## 3. 命令行 / API 参数（与本示例相关）

Python 示例支持：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/yolop/config/yolop.yaml` |
| `--image` | 覆盖输入图片 | YAML 中的 `test_image` |
| `--output` | 输出可视化图片 | `yolop_result.jpg` |
| `--model-path` | 覆盖 YAML 中的模型路径 | 无 |
| `--conf-threshold` | 覆盖置信度阈值 | 使用 YAML 配置 |
| `--iou-threshold` | 覆盖 NMS IoU 阈值 | 使用 YAML 配置 |

C++ 示例的第一个参数必须是配置文件路径，并支持 `--image`、`--output`
和 `--model-path`。

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

**Python：** 在仓库根目录运行：

```bash
python examples/yolop/python/yolop.py --config examples/yolop/config/yolop.yaml
python examples/yolop/python/yolop.py --config examples/yolop/config/yolop.yaml --image /path/to/image.jpg --output yolop_result.jpg
```

**C++：** 在 `build` 目录运行：

```bash
./examples/yolop ../examples/yolop/config/yolop.yaml
./examples/yolop ../examples/yolop/config/yolop.yaml --image /path/to/image.jpg --output yolop_result.jpg
```

## 5. 故障排查

- **模型未找到**：执行 `bash examples/yolop/scripts/download_models.sh`，
  或使用 `--model-path` 指定模型。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或使用
  `--image` 指定图片。
- **模型创建失败**：确认配置中的模型路径、模型输入尺寸和
  `SpaceMITExecutionProvider` 可用。
- **Python 扩展导入失败**：安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。
