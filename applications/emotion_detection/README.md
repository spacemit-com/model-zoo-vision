# 情绪检测应用

对图像进行人脸检测与情绪识别：先检测人脸框，再对每个人脸做情绪分类，输出情绪类别与置信度。

## 1. 模型与权重

- **模型类型**：人脸检测（YOLOv5-Face）+ 情绪分类（ResNet）
- **默认模型文件**：情绪模型 `~/.cache/models/vision/resnet/emotion_resnet50_final.q.onnx`（emotion_model_path）；人脸检测 `~/.cache/models/vision/yolov5-face/yolov5n-face_cut.q.onnx`（face_detector_path）
- **下载**：在本应用目录下执行 `bash scripts/download_models.sh` 会将上述两个模型下载到缓存路径。

**数据（测试图片）**：默认测试图 test_image 指向 `~/.cache/assets/image/001_emotion.jpg`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行应用。

## 2. 配置文件说明（config/emotion_detection.yaml）

本应用使用的配置文件为 `config/emotion_detection.yaml`。与**本应用强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `emotion_config_path` | 情绪模型配置（指向 examples 的 yaml） | `examples/emotion/config/emotion.yaml` |
| `emotion_model_path` | 情绪识别 ONNX 模型路径 | `~/.cache/models/vision/resnet/emotion_resnet50_final.q.onnx` |
| `face_detector_config_path` | 人脸检测配置（指向 examples 的 yaml） | `examples/yolov5-face/config/yolov5-face.yaml` |
| `face_detector_path` | 人脸检测 ONNX 模型路径 | `~/.cache/models/vision/yolov5-face/yolov5n-face_cut.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/001_emotion.jpg` |
| `label_file_path` | 情绪类别标签文件 | `assets/labels/emotion.txt` |
| `image_size` | 情绪模型输入尺寸 [高, 宽] | `[224, 224]` |

## 3. 命令行 / API 参数（与本应用相关）

**Python**（python/example_emotion.py）常用参数：--config、--image、--output、--conf-threshold、--iou-threshold、--num-threads、--face-model-path、--emotion-model-path。

**C++**（需在 cv/build 下先编译）：第一个参数为应用配置路径，图片/输出为位置参数；可选 `--use-camera`、`--camera-id`、`--face-model-path`、`--emotion-model-path`。

## 4. 运行示例

**Python：** 建议在 **cv 组件根目录** 运行：

```bash
python applications/emotion_detection/python/example_emotion.py
python applications/emotion_detection/python/example_emotion.py --config applications/emotion_detection/config/emotion_detection.yaml --image /path/to/face.jpg --output result_emotion.jpg
```

**C++：** 在 `cv/build` 目录下：

```bash
./applications/example_emotion applications/emotion_detection/config/emotion_detection.yaml
./applications/example_emotion applications/emotion_detection/config/emotion_detection.yaml /path/to/face.jpg result_emotion.jpg
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`（在 `applications/emotion_detection/` 下），且 emotion_model_path、face_detector_path 指向的路径存在。
- **测试图片未找到**：默认图片在 ~/.cache/assets/image/ 下。在 cv 根目录执行 bash scripts/download_assets.sh 可下载资源。
- **输入建议**：若使用整张照片，应用会先做人脸检测再对每个人脸做情绪识别；或使用已裁剪的人脸图。
