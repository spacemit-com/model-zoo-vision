# YOLOv8-Pose 示例

姿态估计示例：对图像进行人体关键点检测，输出人体框与关节点坐标。

## 1. 模型与权重

- **模型类型**：姿态估计（YOLOv8-Pose，人体关键点）
- **默认模型文件**：`~/.cache/models/vision/yolov8_pose/yolov8n-pose.q.onnx`（与 `config/yolov8_pose.yaml` 中 `model_path` 一致）
- **下载**：在本示例目录下执行  
  `bash scripts/download_models.sh`  
  会将模型下载到上述缓存路径。

**数据（测试图片）**：默认测试图 `test_image` 指向 `~/.cache/assets/image/013_pose.jpg`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`（并对默认测试图 `013_pose.jpg` 做了单独拉取）；在本示例下执行 `scripts/download_models.sh` 时也会尝试下载该测试图。之后即可直接运行示例。

## 2. 配置文件说明（config/yolov8_pose.yaml）

本示例使用的配置文件为 `config/yolov8_pose.yaml`。与**本模块强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/yolov8_pose/yolov8n-pose.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/013_pose.jpg` |
| `label_file_path` | 类别标签文件（如 COCO） | `assets/labels/coco.txt` |
| ONNX 输入尺寸 | 由模型自动读取，不能通过 YAML 改写 | 见模型文件 |
| `default_params.conf_threshold` | 置信度阈值 | `0.25` |
| `default_params.iou_threshold` | NMS IoU 阈值 | `0.45` |
| `default_params.kp_threshold` | 关键点可见度阈值 | `0.2` |
| `default_params.providers` | ONNX Runtime 执行提供方（单元素列表） | `[SpaceMITExecutionProvider]` |

说明：通用字段（如 `class`、`default_params` 结构）与其它示例一致，不在此重复。

`default_params.kp_threshold` 设置关键点可见度阈值，保留至少 3 个关键点
超过该阈值的人体。旧名称 `point_confidence_threshold` 仍可使用；
同时配置时以 `kp_threshold` 为准。单次请求的 `conf_threshold`、
`iou_threshold`、`kp_threshold` 为正数时覆盖 YAML，`<=0` 使用 YAML 默认值。
`max_det` 通过 `VisionServiceInferParams` 设置，负数不限制、0 返回空结果、
正数限制人体数量，不在本模型的 YAML 中配置。例如已有 `service` 和 `image` 时：

```python
from spacemit_vision import VisionServiceInferParams

params = VisionServiceInferParams()
params.kp_threshold = 0.3
params.max_det = 5
status, response = service.infer(image, params)
```

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/yolov8_pose.py`）常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | 默认 `examples/yolov8_pose/config/yolov8_pose.yaml` |
| `--image` | 输入图片路径 | 使用 yaml 中 `test_image` |
| `--output` | 输出图片路径 | `yolov8_pose_result.jpg` |
| `--use-camera` | 使用摄像头输入 | 关 |
| `--camera-id` | 摄像头设备 ID | `0` |
| `--model-path` | 覆盖 yaml 中的 `model_path` | 无 |

**C++ 示例**（需在 `cv/build` 下先编译）：

- 第一个参数：配置文件路径，如 `../examples/yolov8_pose/config/yolov8_pose.yaml`
- 可选：`--image`、`--output`、`--use-camera`、`--camera-id`、`--model-path`

## 4. 运行示例

**Python：**

```bash
cd examples/yolov8_pose/python
python yolov8_pose.py --config ../config/yolov8_pose.yaml
python yolov8_pose.py --config ../config/yolov8_pose.yaml --image /path/to/image.jpg --output result.jpg
python yolov8_pose.py --config ../config/yolov8_pose.yaml --use-camera
```

**C++：** 在 `cv/build` 目录下：

```bash
./examples/yolov8_pose ../examples/yolov8_pose/config/yolov8_pose.yaml
./examples/yolov8_pose ../examples/yolov8_pose/config/yolov8_pose.yaml --image /path/to/image.jpg --output result.jpg
./examples/yolov8_pose ../examples/yolov8_pose/config/yolov8_pose.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/yolov8_pose/yolov8n-pose.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。
