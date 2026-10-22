# YOLOv8-Seg 示例

实例分割示例：对图像进行目标检测与实例分割，输出检测框与分割掩码。

## 1. 模型与权重

- **模型类型**：实例分割（YOLOv8-Seg）
- **默认模型文件**：`~/.cache/models/vision/yolov8_seg/yolov8n-seg.q.onnx`（与 `config/yolov8_seg.yaml` 中 `model_path` 一致）
- **下载**：在仓库根目录执行 `bash examples/yolov8_seg/scripts/download_models.sh` 会将模型下载到上述缓存路径。

**数据（测试图片）**：默认测试图 `test_image` 指向 `~/.cache/assets/image/006_test.jpg`。若尚未下载资源，请在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/yolov8_seg.yaml）

本示例使用的配置文件为 `config/yolov8_seg.yaml`。与**本示例强相关**的字段：`model_path`、`test_image`、`label_file_path`、`default_params.conf_threshold`、`default_params.iou_threshold`、`default_params.providers` 等，含义与 YOLOv8 检测类似。

单次请求的正数 `conf_threshold`、`iou_threshold` 覆盖 YAML 默认值，
`<=0` 使用配置默认值。`mask_threshold`、`max_det` 通过
`VisionServiceInferParams` 设置，不在本模型的 YAML 中配置：
`mask_threshold` 有效范围为 `(0,1]`，`<=0` 使用默认值 `0.5`；
`max_det` 为负数时不限制，0 返回空结果，正数限制实例数量。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/yolov8_seg.py`）常用参数：`--config`、`--image`、`--output`、`--use-camera`、`--camera-id`、`--model-path`。

**C++ 示例**：第一个参数为配置文件路径，可选 `--image`、`--output`、`--use-camera`、`--camera-id`、`--model-path`。

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

摄像头模式需要可用的摄像头设备和 OpenCV 图形显示环境；无桌面的 SSH 会话可先运行图片示例。

**Python：** 在仓库根目录运行：

```bash
python examples/yolov8_seg/python/yolov8_seg.py --config examples/yolov8_seg/config/yolov8_seg.yaml
python examples/yolov8_seg/python/yolov8_seg.py --config examples/yolov8_seg/config/yolov8_seg.yaml --image /path/to/image.jpg --output result.jpg
python examples/yolov8_seg/python/yolov8_seg.py --config examples/yolov8_seg/config/yolov8_seg.yaml --use-camera
```

**C++：** 在 `build` 目录运行：

```bash
./examples/yolov8_seg ../examples/yolov8_seg/config/yolov8_seg.yaml
./examples/yolov8_seg ../examples/yolov8_seg/config/yolov8_seg.yaml --image /path/to/image.jpg --output result.jpg
./examples/yolov8_seg ../examples/yolov8_seg/config/yolov8_seg.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `bash examples/yolov8_seg/scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/yolov8_seg/yolov8n-seg.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在仓库根目录执行 `bash scripts/download_assets.sh` 可下载资源。
