# YOLO26 示例

目标检测示例，仅支持带 NMS 的 e2e 输出 `[N, 6]` 或 `[1, N, 6]`，每行是 `x1,y1,x2,y2,score,class`。示例输出检测框、类别和分数，并保存绘制结果。

## 1. 模型与权重

- **模型类型**：目标检测（YOLO26）
- **默认模型文件**：`~/.cache/models/vision/yolo26/yolo26n.q.onnx`，与 `config/yolo26.yaml` 一致。
- **下载**：在仓库根目录执行：

```bash
bash examples/yolo26/scripts/download_models.sh
```

该示例使用已经在模型内完成 NMS 的输出，不支持直接替换成其他未解码的检测头。

**测试图片**：默认使用 `~/.cache/assets/image/006_test.jpg`。资源未下载时，在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/yolo26.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/yolo26/yolo26n.q.onnx` |
| `test_image` | 默认测试图片 | `~/.cache/assets/image/006_test.jpg` |
| `label_file_path` | 类别标签文件 | `assets/labels/coco.txt` |
| `default_params.conf_threshold` | 置信度阈值 | `0.25` |
| `default_params.iou_threshold` | 保留的 IoU 参数；e2e 输出已含 NMS，不再执行额外 NMS | `0.45` |
| `default_params.num_threads` | 推理线程数 | `8` |
| `default_params.providers` | 执行提供方，单元素列表 | `[SpaceMITExecutionProvider]` |
| `default_params.preprocess.backend` | 预处理策略：`cpu` / `auto` / `opencl` / `v2d` | `auto` |
| `camera.backend` | C++ 摄像头采集方式，与预处理策略独立 | `opencv` |
| `camera.device` | C++ 摄像头设备 | `/dev/video0` |
| `camera.width` / `camera.height` / `camera.fps` | C++ 摄像头采集尺寸与帧率 | `1280` / `720` / `30` |

ONNX 输入尺寸由模型读取。`auto` 对普通 BGR 图片使用 CPU，对 NV12 DMA-BUF
尝试可用且兼容的 V2D、OpenCL 后端，再回退 CPU；硬件后端需启用相应构建选项。
显式 `opencl` / `v2d` 默认失败即报错，配置 `default_params.preprocess.fallback: cpu`
可允许回退。摄像头配置仅在 C++ 示例传入 `--use-camera` 时使用。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/yolo26.py`）：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/yolo26/config/yolo26.yaml` |
| `--image` | 输入图片 | YAML 中的 `test_image` |
| `--output` | 输出图片 | `yolo26_result.jpg` |
| `--conf-threshold` / `--iou-threshold` | 本次调用的阈值覆盖 | YAML 默认值 |
| `--use-camera` / `--camera-id` | 使用摄像头及设备 ID | 关闭 / `0` |
| `--model-path` | 覆盖主模型路径 | 无 |

**C++ 示例**：第一个参数是配置文件路径，支持 `--image`、`--output`、
`--use-camera`、`--camera-id`、`--model-path`，以及可选 MPP 采集参数。
C++ 示例没有 `--conf-threshold` / `--iou-threshold` 选项，可在 YAML 中设置。

API 可通过 `VisionServiceInferParams` 设置置信度、IoU 和最大检测数：

```python
from spacemit_vision import VisionServiceInferParams

params = VisionServiceInferParams()
params.conf_threshold = 0.3
params.max_det = 20
status, results = service.infer_image(image, params=params)
```

置信度和 IoU 阈值必须有限且不大于 1；`<=0` 使用模型默认值。
`max_det` 负数不限制数量，0 返回空结果，正数限制结果数；不要将其写入本模型 YAML。

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

摄像头模式需要可用的摄像头设备和 OpenCV 图形显示环境；无桌面的 SSH 会话可先运行图片示例。

**Python：** 在仓库根目录运行：

```bash
python examples/yolo26/python/yolo26.py --config examples/yolo26/config/yolo26.yaml
python examples/yolo26/python/yolo26.py --config examples/yolo26/config/yolo26.yaml --image /path/to/image.jpg --output result.jpg
python examples/yolo26/python/yolo26.py --config examples/yolo26/config/yolo26.yaml --use-camera --camera-id 0
```

**C++：** 在 `build` 目录运行：

```bash
./examples/yolo26 ../examples/yolo26/config/yolo26.yaml
./examples/yolo26 ../examples/yolo26/config/yolo26.yaml --image /path/to/image.jpg --output result.jpg
./examples/yolo26 ../examples/yolo26/config/yolo26.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：执行 `bash examples/yolo26/scripts/download_models.sh`，并核对 `model_path`。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或用 `--image` 指定图片。
- **输出格式不支持**：核对所用权重是否符合上面的导出格式。
- **硬件预处理不可用**：检查构建选项和输入类型，或使用 `cpu` / `auto`。
