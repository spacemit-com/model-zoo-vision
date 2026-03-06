# 区域闯入识别应用

对视频/摄像头进行多目标检测与跟踪（ByteTrack），并判断目标是否闯入指定区域（ROI）。支持用 3 个或 4 个点（顺时针）定义封闭多边形区域；可选开启“进入计数”模式，按 track_id 的 outside→inside 变化统计进入次数。

## 1. 模型与权重

- **模型类型**：多目标跟踪（YOLOv8 检测 + ByteTrack）
- **默认模型文件**：`~/.cache/models/vision/yolov8/yolov8n.q.onnx`（与 `config/intrusion_detection.yaml` 中 `tracker_model_path` 一致）
- **下载**：在本应用目录下执行  
  `bash scripts/download_models.sh`  
  会将模型下载到上述缓存路径。

**数据（测试视频）**：默认测试视频 `test_video` 指向 `~/.cache/assets/video/001_crowd.mp4`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行应用。

## 2. 配置文件说明（config/intrusion_detection.yaml）

本应用使用的配置文件为 `config/intrusion_detection.yaml`。与**本应用强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `tracker_config_path` | 跟踪器配置（指向 examples 的 yaml） | `examples/bytetrack/config/bytetrack.yaml` |
| `tracker_model_path` | 检测/跟踪 ONNX 模型路径 | `~/.cache/models/vision/yolov8/yolov8n.q.onnx` |
| `test_video` | 默认测试视频路径 | `~/.cache/assets/video/001_crowd.mp4` |
| `label_file_path` | 检测类别标签文件（用于判断“人”及显示） | `assets/labels/coco.txt` |
| `image_size` | 模型输入尺寸 [高, 宽] | `[640, 640]` |

说明：ROI 区域由命令行 `--roi-points` 指定（如 3 个或 4 个点，顺时针）；未指定时 Python 使用默认中心区域。

## 3. 命令行 / API 参数（与本应用相关）

**Python**（`python/example_intrusion_identification.py`）常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 应用配置 yaml 路径 | `applications/intrusion_detection/config/intrusion_detection.yaml` |
| `--video` | 输入视频路径 | 使用 yaml 中 `test_video` |
| `--use-camera` | 使用摄像头 | 关 |
| `--camera-id` | 摄像头设备 ID | 无 |
| `--conf-threshold` | 检测置信度阈值 | 使用 tracker 配置 |
| `--iou-threshold` | NMS IoU 阈值 | 使用 tracker 配置 |
| `--frame-rate` | 视频帧率（用于跟踪等） | 使用 tracker 配置 |
| `--num-threads` | 推理线程数 | 使用配置 |
| `--model-path` | 跟踪/检测模型 ONNX 路径，覆盖 yaml 中 `tracker_model_path` | 无 |
| `--roi-points` | ROI 顶点（顺时针），例如 `x1,y1 x2,y2 x3,y3` 或 4 点 | 无 |
| `--counting-mode` | 开启后统计“进入区域”次数（按 track_id outside→inside） | 关 |

**C++**（需在 `cv/build` 下先编译）：第一个参数为应用配置路径，可选 `--config`、`--video`、`--use-camera`、`--camera-id`、`--model-path`。

## 4. 运行示例

**Python：** 建议在 **cv 组件根目录** 运行：

```bash
python applications/intrusion_detection/python/example_intrusion_identification.py
python applications/intrusion_detection/python/example_intrusion_identification.py --config applications/intrusion_detection/config/intrusion_detection.yaml --video /path/to/video.mp4
python applications/intrusion_detection/python/example_intrusion_identification.py --config applications/intrusion_detection/config/intrusion_detection.yaml --use-camera --camera-id 0
# 指定 ROI 三点（顺时针）并开启进入计数
python applications/intrusion_detection/python/example_intrusion_identification.py --roi-points 100,200 300,200 320,400 --counting-mode --video /path/to/video.mp4
```

**C++：** 在 `cv/build` 目录下：

```bash
./applications/example_intrusion_identification applications/intrusion_detection/config/intrusion_detection.yaml
./applications/example_intrusion_identification applications/intrusion_detection/config/intrusion_detection.yaml --video /path/to/video.mp4
./applications/example_intrusion_identification applications/intrusion_detection/config/intrusion_detection.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`（在 `applications/intrusion_detection/` 下），且 `tracker_model_path` 指向 `~/.cache/models/vision/yolov8/yolov8n.q.onnx`。
- **测试视频未找到**：默认视频在 `~/.cache/assets/video/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。
- **ROI 未生效**：检查 `--roi-points` 格式为 `x,y` 且至少 3 个点、顺时针顺序。
