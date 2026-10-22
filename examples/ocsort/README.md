# OC-SORT 示例

多目标跟踪示例：对视频进行检测 + 跟踪，输出带 track ID 的检测框（YOLOv8 检测 + OC-SORT 跟踪）。

## 1. 模型与权重

- **模型类型**：多目标跟踪（YOLOv8 检测 + OC-SORT）
- **默认模型文件**：`~/.cache/models/vision/ocsort/yolov8n_no_dfl.q.onnx`（与 `config/ocsort.yaml` 中 `model_path` 一致）
- **下载**：在仓库根目录执行：

```bash
bash examples/ocsort/scripts/download_models.sh
```

会将模型下载到上述缓存路径。

**数据（测试视频）**：默认测试视频 `test_video` 指向 `~/.cache/assets/video/003_palace.mp4`。若尚未下载资源，请在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/ocsort.yaml）

本示例使用的配置文件为 `config/ocsort.yaml`。与**本示例强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | 检测器 ONNX 模型路径 | `~/.cache/models/vision/ocsort/yolov8n_no_dfl.q.onnx` |
| `test_video` | 默认测试视频路径 | `~/.cache/assets/video/003_palace.mp4` |
| `label_file_path` | 类别标签文件（如 COCO） | `assets/labels/coco.txt` |
| `default_params.conf_threshold` | 检测置信度阈值 | `0.25` |
| `default_params.iou_threshold` | NMS IoU 阈值 | `0.45` |
| `default_params.track_thresh` | 跟踪置信度阈值 | `0.6` |
| `default_params.track_buffer` | 跟踪缓冲帧数 | `60` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `[SpaceMITExecutionProvider]` |

说明：通用字段（如 `class`、`default_params` 结构）与其他示例一致，不在此重复。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/ocsort.py`）常用参数：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | 默认 `examples/ocsort/config/ocsort.yaml` |
| `--video` | 输入视频路径 | 使用 YAML 中 `test_video` |
| `--use-camera` | 使用摄像头输入 | 关闭 |
| `--camera-id` | 摄像头设备 ID | `0` |
| `--model-path` | 覆盖 YAML 中的 `model_path` | 无 |

**C++ 示例**：

- 第一个参数：配置文件路径，如 `../examples/ocsort/config/ocsort.yaml`
- 可选：`--video`、`--use-camera`、`--camera-id`、`--model-path`

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

视频和摄像头示例会调用 OpenCV 窗口显示结果，需要可用的图形会话；
仅有 SSH、没有显示服务的环境不能直接运行这些交互命令。

**Python：** 在仓库根目录运行：

```bash
python examples/ocsort/python/ocsort.py --config examples/ocsort/config/ocsort.yaml
python examples/ocsort/python/ocsort.py --config examples/ocsort/config/ocsort.yaml --video /path/to/video.mp4
python examples/ocsort/python/ocsort.py --config examples/ocsort/config/ocsort.yaml --use-camera
```

**C++：** 在 `build` 目录运行：

```bash
./examples/ocsort ../examples/ocsort/config/ocsort.yaml
./examples/ocsort ../examples/ocsort/config/ocsort.yaml --video /path/to/video.mp4
./examples/ocsort ../examples/ocsort/config/ocsort.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `bash examples/ocsort/scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/ocsort/yolov8n_no_dfl.q.onnx`。
- **测试视频未找到**：默认视频在 `~/.cache/assets/video/` 下。在仓库根目录执行 `bash scripts/download_assets.sh` 可下载资源。
