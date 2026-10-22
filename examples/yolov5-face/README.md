# YOLOv5-Face 示例

人脸检测示例：对图像进行人脸框检测，输出人脸边界框（不包含人脸识别/相似度）。与 ArcFace 示例（人脸识别）区分：本示例仅做检测。

## 1. 模型与权重

- **模型类型**：人脸检测（YOLOv5-Face，仅检测框）
- **默认模型文件**：`~/.cache/models/vision/yolov5-face/yolov5n-face_cut.q.onnx`（与 `config/yolov5-face.yaml` 中 `model_path` 一致）
- **下载**：在仓库根目录执行：

```bash
bash examples/yolov5-face/scripts/download_models.sh
```

会将模型下载到上述缓存路径。

**数据（测试图片）**：默认测试图 `test_image` 指向 `~/.cache/assets/image/006_test.jpg`。若尚未下载资源，请在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/yolov5-face.yaml）

本示例使用的配置文件为 `config/yolov5-face.yaml`。与**本示例强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/yolov5-face/yolov5n-face_cut.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/006_test.jpg` |
| `default_params.conf_threshold` | 置信度阈值 | `0.25` |
| `default_params.iou_threshold` | NMS IoU 阈值 | `0.45` |
| `default_params.providers` | ONNX Runtime 执行提供方（单元素列表） | `[SpaceMITExecutionProvider]` |

说明：通用字段（如 `class`、`default_params` 结构）与其他示例一致，不在此重复。

YAML 使用 `default_params.conf_threshold` 和 `default_params.iou_threshold`。
旧名称 `conf_thres`、`iou_thres` 仍可使用；新旧名称同时出现时以统一名称为准。
Python 示例的 `--conf-threshold`、`--iou-threshold` 仅覆盖本次运行；
API 中对应的请求参数为正数时覆盖 YAML，`<=0` 使用配置默认值，例如
`service.infer_image(image, conf=0.3, iou=0.4)`。`max_det` 仅通过
`VisionServiceInferParams` 设置，负数不限制、0 返回空结果、正数限制框数量。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/yolov5_face.py`）常用参数：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | 默认 `examples/yolov5-face/config/yolov5-face.yaml` |
| `--image` | 输入图片路径 | 使用 YAML 中 `test_image` |
| `--output` | 输出图片路径 | `result_face.jpg` |
| `--use-camera` | 使用摄像头输入 | 关闭 |
| `--camera-id` | 摄像头设备 ID | `0` |
| `--model-path` | 覆盖 YAML 中的 `model_path` | 无 |

**C++ 示例**：

- 第一个参数：配置文件路径，如 `../examples/yolov5-face/config/yolov5-face.yaml`
- 可选：`--image`、`--output`、`--use-camera`、`--camera-id`、`--model-path`

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

摄像头模式需要可用的摄像头设备和 OpenCV 图形显示环境；无桌面的 SSH 会话可先运行图片示例。

**Python：** 在仓库根目录运行：

```bash
python examples/yolov5-face/python/yolov5_face.py --config examples/yolov5-face/config/yolov5-face.yaml
python examples/yolov5-face/python/yolov5_face.py --config examples/yolov5-face/config/yolov5-face.yaml --image /path/to/image.jpg --output result_face.jpg
python examples/yolov5-face/python/yolov5_face.py --config examples/yolov5-face/config/yolov5-face.yaml --use-camera
```

**C++：** 在 `build` 目录运行：

```bash
./examples/yolov5-face ../examples/yolov5-face/config/yolov5-face.yaml
./examples/yolov5-face ../examples/yolov5-face/config/yolov5-face.yaml --image /path/to/image.jpg --output result_face.jpg
./examples/yolov5-face ../examples/yolov5-face/config/yolov5-face.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `bash examples/yolov5-face/scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/yolov5-face/yolov5n-face_cut.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在仓库根目录执行 `bash scripts/download_assets.sh` 可下载资源。
