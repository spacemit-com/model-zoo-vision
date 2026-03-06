# ByteTrack 示例

多目标跟踪示例：对视频进行检测 + 跟踪，输出带 track ID 的检测框（YOLOv8 检测 + ByteTrack 跟踪）。

## 1. 模型与权重

- **模型类型**：多目标跟踪（YOLOv8 检测 + ByteTrack）
- **默认模型文件**：`~/.cache/models/vision/yolov8/yolov8n.q.onnx`（与 `config/bytetrack.yaml` 中 `model_path` 一致）
- **下载**：在本示例目录下执行  
  `bash scripts/download_models.sh`  
  会将模型下载到上述缓存路径。

**数据（测试视频）**：默认测试视频 `test_video` 指向 `~/.cache/assets/video/003_palace.mp4`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/bytetrack.yaml）

本示例使用的配置文件为 `config/bytetrack.yaml`。与**本模块强相关**的字段：`model_path`、`test_video`、`label_file_path`、`default_params.conf_threshold`、`default_params.iou_threshold`、`default_params.track_buffer`、`default_params.frame_rate`、`default_params.providers` 等。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/bytetrack.py`）常用参数：`--config`、`--video`、`--use-camera`、`--camera-id`、`--model-path`。

**C++ 示例**（需在 `cv/build` 下先编译）：第一个参数为配置文件路径，可选 `--video`、`--use-camera`、`--camera-id`、`--model-path`。

## 4. 运行示例

**Python：**

```bash
cd examples/bytetrack/python
python bytetrack.py --config ../config/bytetrack.yaml
python bytetrack.py --config ../config/bytetrack.yaml --video /path/to/video.mp4
python bytetrack.py --config ../config/bytetrack.yaml --use-camera
```

**C++：** 在 `cv/build` 目录下：

```bash
./examples/bytetrack examples/bytetrack/config/bytetrack.yaml
./examples/bytetrack examples/bytetrack/config/bytetrack.yaml --video /path/to/video.mp4
./examples/bytetrack examples/bytetrack/config/bytetrack.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/yolov8/yolov8n.q.onnx`。
- **测试视频未找到**：默认视频在 `~/.cache/assets/video/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。

