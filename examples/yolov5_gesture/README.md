# YOLOv5 手势检测示例

手势检测示例：对图像进行手势类别检测，输出检测框与手势类别标签。

## 1. 模型与权重

- **模型类型**：目标检测（YOLOv5 手势）
- **默认模型文件**：`~/.cache/models/vision/yolov5/yolov5_gesture.q.onnx`（与 `config/yolov5_gesture.yaml` 中 `model_path` 一致）
- **下载**：在仓库根目录执行：

```bash
bash examples/yolov5_gesture/scripts/download_models.sh
```

会将模型下载到上述缓存路径。

**数据（测试图片）**：默认测试图 `test_image` 指向 `~/.cache/assets/image/012_gesture.jpg`。若尚未下载资源，请在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/yolov5_gesture.yaml）

本示例使用的配置文件为 `config/yolov5_gesture.yaml`。与**本示例强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/yolov5/yolov5_gesture.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/012_gesture.jpg` |
| `label_file_path` | 手势类别标签文件 | `assets/labels/gesture.txt` |
| ONNX 输入尺寸 | 由模型自动读取，不能通过 YAML 改写 | 见模型文件 |
| `default_params.conf_threshold` | 置信度阈值 | `0.8` |
| `default_params.iou_threshold` | NMS IoU 阈值 | `0.5` |
| `default_params.providers` | ONNX Runtime 执行提供方（单元素列表） | `[SpaceMITExecutionProvider]` |

说明：通用字段（如 `class`、`default_params` 结构）与其他示例一致，不在此重复。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/yolov5_gesture.py`）常用参数：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | 默认 `examples/yolov5_gesture/config/yolov5_gesture.yaml` |
| `--image` | 输入图片路径 | 使用 YAML 中 `test_image` |
| `--output` | 输出图片路径 | `result_gesture.jpg` |
| `--use-camera` | 使用摄像头输入 | 关闭 |
| `--camera-id` | 摄像头设备 ID | `0` |
| `--model-path` | 覆盖 YAML 中的 `model_path` | 无 |

**C++ 示例**：

- 第一个参数：配置文件路径，如 `../examples/yolov5_gesture/config/yolov5_gesture.yaml`
- 可选：`--image`、`--output`、`--use-camera`、`--camera-id`、`--model-path`

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

摄像头模式需要可用的摄像头设备和 OpenCV 图形显示环境；无桌面的 SSH 会话可先运行图片示例。

**Python：** 在仓库根目录运行：

```bash
python examples/yolov5_gesture/python/yolov5_gesture.py --config examples/yolov5_gesture/config/yolov5_gesture.yaml
python examples/yolov5_gesture/python/yolov5_gesture.py --config examples/yolov5_gesture/config/yolov5_gesture.yaml --image /path/to/image.jpg --output result.jpg
python examples/yolov5_gesture/python/yolov5_gesture.py --config examples/yolov5_gesture/config/yolov5_gesture.yaml --use-camera
```

**C++：** 在 `build` 目录运行：

```bash
./examples/yolov5_gesture ../examples/yolov5_gesture/config/yolov5_gesture.yaml
./examples/yolov5_gesture ../examples/yolov5_gesture/config/yolov5_gesture.yaml --image /path/to/image.jpg --output result.jpg
./examples/yolov5_gesture ../examples/yolov5_gesture/config/yolov5_gesture.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `bash examples/yolov5_gesture/scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/yolov5/yolov5_gesture.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在仓库根目录执行 `bash scripts/download_assets.sh` 可下载资源。
