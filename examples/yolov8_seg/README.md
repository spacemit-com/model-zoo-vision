# YOLOv8-Seg 示例

实例分割示例：对图像进行目标检测与实例分割，输出检测框与分割掩码。

## 1. 模型与权重

- **模型类型**：实例分割（YOLOv8-Seg）
- **默认模型文件**：`~/.cache/models/vision/yolov8_seg/yolov8n-seg.q.onnx`（与 `config/yolov8_seg.yaml` 中 `model_path` 一致）
- **下载**：在本示例目录下执行 `bash scripts/download_models.sh` 会将模型下载到上述缓存路径。

**数据（测试图片）**：默认测试图 `test_image` 指向 `~/.cache/assets/image/006_test.jpg`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/yolov8_seg.yaml）

本示例使用的配置文件为 `config/yolov8_seg.yaml`。与**本模块强相关**的字段：`model_path`、`test_image`、`label_file_path`、`image_size`、`default_params.conf_threshold`、`default_params.iou_threshold`、`default_params.providers` 等，含义与 YOLOv8 检测类似。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/yolov8_seg.py`）常用参数：`--config`、`--image`、`--output`、`--use-camera`、`--camera-id`、`--model-path`。

**C++ 示例**（需在 `cv/build` 下先编译）：第一个参数为配置文件路径，可选 `--image`、`--output`、`--use-camera`、`--camera-id`、`--model-path`。

## 4. 运行示例

**Python：**

```bash
cd examples/yolov8_seg/python
python yolov8_seg.py --config ../config/yolov8_seg.yaml
python yolov8_seg.py --config ../config/yolov8_seg.yaml --image /path/to/image.jpg --output result.jpg
python yolov8_seg.py --config ../config/yolov8_seg.yaml --use-camera
```

**C++：** 在 `cv/build` 目录下：

```bash
./examples/yolov8_seg examples/yolov8_seg/config/yolov8_seg.yaml
./examples/yolov8_seg examples/yolov8_seg/config/yolov8_seg.yaml --image /path/to/image.jpg --output result.jpg
./examples/yolov8_seg examples/yolov8_seg/config/yolov8_seg.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/yolov8_seg/yolov8n-seg.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。

