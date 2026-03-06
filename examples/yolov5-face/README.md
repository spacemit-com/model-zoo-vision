# YOLOv5-Face 示例

人脸检测示例：对图像进行人脸框检测，输出人脸边界框（不包含人脸识别/相似度）。与 ArcFace 示例（人脸识别）区分：本示例仅做检测。

## 1. 模型与权重

- **模型类型**：人脸检测（YOLOv5-Face，仅检测框）
- **默认模型文件**：`~/.cache/models/vision/yolov5-face/yolov5n-face_cut.q.onnx`（与 `config/yolov5-face.yaml` 中 `model_path` 一致）
- **下载**：在本示例目录下执行  
  `bash scripts/download_models.sh`  
  会将模型下载到上述缓存路径。

**数据（测试图片）**：默认测试图 `test_image` 指向 `~/.cache/assets/image/006_test.jpg`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/yolov5-face.yaml）

本示例使用的配置文件为 `config/yolov5-face.yaml`。与**本模块强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/yolov5-face/yolov5n-face_cut.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/006_test.jpg` |
| `default_params.conf_thres` / `conf_threshold` | 置信度阈值 | `0.25` |
| `default_params.iou_thres` / `iou_threshold` | NMS IoU 阈值 | `0.45` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

说明：通用字段（如 `class`、`default_params` 结构）与其它示例一致，不在此重复。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/yolov5_face.py`）常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | 默认 `examples/yolov5-face/config/yolov5-face.yaml` |
| `--image` | 输入图片路径 | 使用 yaml 中 `test_image` |
| `--output` | 输出图片路径 | `result_face.jpg` |
| `--use-camera` | 使用摄像头输入 | 关 |
| `--camera-id` | 摄像头设备 ID | `0` |
| `--model-path` | 覆盖 yaml 中的 `model_path` | 无 |

**C++ 示例**（需在 `cv/build` 下先编译）：

- 第一个参数：配置文件路径，如 `examples/yolov5-face/config/yolov5-face.yaml`
- 可选：`--image`、`--output`、`--use-camera`、`--camera-id`、`--model-path`

## 4. 运行示例

**Python：**

```bash
cd examples/yolov5-face/python
python yolov5_face.py --config ../config/yolov5-face.yaml
python yolov5_face.py --config ../config/yolov5-face.yaml --image /path/to/image.jpg --output result_face.jpg
python yolov5_face.py --config ../config/yolov5-face.yaml --use-camera
```

**C++：** 在 `cv/build` 目录下：

```bash
./examples/yolov5-face examples/yolov5-face/config/yolov5-face.yaml
./examples/yolov5-face examples/yolov5-face/config/yolov5-face.yaml --image /path/to/image.jpg --output result_face.jpg
./examples/yolov5-face examples/yolov5-face/config/yolov5-face.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/yolov5-face/yolov5n-face_cut.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。
