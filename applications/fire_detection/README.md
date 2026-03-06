# 火焰检测应用

使用 YOLOv8 火焰检测模型对图像/视频/摄像头进行火焰检测，输出检测框与类别标签。

## 1. 模型与权重

- **模型类型**：目标检测（YOLOv8 火焰专用）
- **默认模型文件**：`~/.cache/models/vision/yolov8/yolov8_fire.q.onnx`（与 `config/fire_detection.yaml` 中 `detector_model_path` 一致）
- **下载**：在本应用目录下执行  
  `bash scripts/download_models.sh`  
  会将模型下载到上述缓存路径。

**数据（测试图片/视频）**：默认测试图 `test_image` 指向 `~/.cache/assets/image/002_fire.jpg`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行应用。

## 2. 配置文件说明（config/fire_detection.yaml）

本应用使用的配置文件为 `config/fire_detection.yaml`。与**本应用强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `detector_config_path` | 检测器配置（指向 examples 的 yaml） | `examples/yolov8/config/yolov8.yaml` |
| `detector_model_path` | 火焰检测 ONNX 模型路径 | `~/.cache/models/vision/yolov8/yolov8_fire.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/002_fire.jpg` |
| `label_file_path` | 类别标签文件（如 fire） | `assets/labels/fire.txt` |
| `image_size` | 输入尺寸 [高, 宽] | `[640, 640]` |

说明：检测器内部参数（如 `conf_threshold`、`iou_threshold`、`providers`）由 `detector_config_path` 指向的 yaml 及命令行覆盖。

## 3. 命令行 / API 参数（与本应用相关）

**Python**（`python/example_fire_detection.py`）常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 应用配置 yaml 路径 | `applications/fire_detection/config/fire_detection.yaml` |
| `--image` | 输入图片路径（与 --video/--use-camera 二选一） | 使用 yaml 中 `test_image` |
| `--video` | 输入视频路径 | 无 |
| `--use-camera` | 使用摄像头 | 关 |
| `--camera-id` | 摄像头设备 ID | 无 |
| `--output` | 输出图片/视频路径 | 图片模式默认 `output_fire_detection.jpg` |
| `--conf-threshold` | 置信度阈值 | 使用 detector 配置 |
| `--iou-threshold` | NMS IoU 阈值 | 使用 detector 配置 |
| `--num-threads` | 推理线程数 | 使用 detector 配置 |
| `--model-path` | 检测模型 ONNX 路径，覆盖 yaml 中 `detector_model_path` | 无 |

**C++**（需在 `cv/build` 下先编译）：第一个参数为应用配置路径，图片路径与输出路径为**位置参数**；可选 `--use-camera`、`--camera-id`、`--model-path`。C++ 不支持视频输入（仅图片或摄像头）。

## 4. 运行示例

**Python：** 建议在 **cv 组件根目录** 运行，以便默认配置路径生效：

```bash
# 使用默认配置与 test_image
python applications/fire_detection/python/example_fire_detection.py
python applications/fire_detection/python/example_fire_detection.py --config applications/fire_detection/config/fire_detection.yaml --image /path/to/image.jpg --output result.jpg
python applications/fire_detection/python/example_fire_detection.py --config applications/fire_detection/config/fire_detection.yaml --video /path/to/video.mp4
python applications/fire_detection/python/example_fire_detection.py --config applications/fire_detection/config/fire_detection.yaml --use-camera
```

**C++：** 在 `cv/build` 目录下：

```bash
./applications/example_fire_detection applications/fire_detection/config/fire_detection.yaml
./applications/example_fire_detection applications/fire_detection/config/fire_detection.yaml /path/to/image.jpg result.jpg
./applications/example_fire_detection applications/fire_detection/config/fire_detection.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`（在 `applications/fire_detection/` 下），且 `detector_model_path` 指向 `~/.cache/models/vision/yolov8/yolov8_fire.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。
