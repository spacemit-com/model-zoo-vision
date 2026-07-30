# YOLOv8 示例

目标检测示例：对图像进行多类别目标检测，输出检测框与类别标签。

## 1. 模型与权重

- **模型类型**：目标检测（YOLOv8）
- **默认模型文件**：`~/.cache/models/vision/yolov8/yolov8n_no_dfl.q.onnx`（与 `config/yolov8.yaml` 中 `model_path` 一致）
- **下载**：在本示例目录下执行  
  `bash scripts/download_models.sh`  
  会将模型下载到上述缓存路径。

### 模型格式说明

本示例的检测器**同时支持两种 ONNX 导出格式**，运行时按模型输出数量自动选择后处理路径，无需额外配置：

| 模型文件 | 导出格式 | ONNX 输出 | 后处理路径 |
|----------|----------|-----------|------------|
| `yolov8n_no_dfl.q.onnx`（默认）| 多分支头，DFL 解码层未内置、留给后处理端 | 多个（box/score/score_sum 每尺度一组）| 多分支 DFL 解码 |
| `yolov8n.q.onnx` | 官方 Ultralytics 单输出（含内置解码）| 1 个，形如 `[1, 84, 8400]` | 单输出解析 |

`s` / `m` 规格同理。文件名中的 `_no_dfl` 指模型图内**未内置** DFL 解码层，因此导出为多分支裸输出，由后处理端完成 DFL 解码（即多分支路径）；无后缀文件则是官方导出的单输出格式，解码已内置在模型图内。`download_models.sh` 会同时下载两套模型，便于对比与回归测试；默认 `model_path` 指向 `_no_dfl` 多分支模型。切换到官方单输出模型时，将 `config/yolov8.yaml` 的 `model_path` 改为对应文件名（或用 `--model-path` 覆盖）即可，代码会按输出数量自动切换后处理路径。

**数据（测试图片）**：默认测试图 `test_image` 指向 `~/.cache/assets/image/006_test.jpg`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/yolov8.yaml）

本示例使用的配置文件为 `config/yolov8.yaml`。与**本模块强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/yolov8/yolov8n_no_dfl.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/006_test.jpg` |
| `label_file_path` | 类别标签文件（如 COCO） | `assets/labels/coco.txt` |
| `image_size` | 输入尺寸 [宽, 高] | `[640, 640]` |
| `default_params.conf_threshold` | 置信度阈值 | `0.25` |
| `default_params.iou_threshold` | NMS IoU 阈值 | `0.45` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |
| `default_params.preprocess.backend` | 预处理策略：`cpu` / `auto` / `opencl` | `auto` |

说明：通用字段（如 `class`、`default_params` 结构）与其它示例一致，不在此重复。

`auto` 不与 MPP 绑定：普通 BGR 图片仍走 CPU；当调用方提供
`NV12 + DMA-BUF` 时尝试 OpenCL。显式 `opencl` 用于强制验证，
只接受 `NV12 + DMA-BUF`，OpenCL 不可用或输入不兼容时均不会回退。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/yolov8.py`）常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | 默认 `examples/yolov8/config/yolov8.yaml` |
| `--image` | 输入图片路径 | 使用 yaml 中 `test_image` |
| `--output` | 输出图片路径 | `result.jpg` |
| `--use-camera` | 使用摄像头输入 | 关 |
| `--camera-id` | 摄像头设备 ID | `0` |
| `--model-path` | 覆盖 yaml 中的 `model_path` | 无 |

**C++ 示例**（需在 `cv/build` 下先编译）：

- 第一个参数：配置文件路径，如 `examples/yolov8/config/yolov8.yaml`
- 可选：`--image`、`--output`、`--use-camera`、`--camera-id`、`--model-path`

## 4. 运行示例

**Python：**

```bash
cd examples/yolov8/python
python yolov8.py --config ../config/yolov8.yaml
python yolov8.py --config ../config/yolov8.yaml --image /path/to/image.jpg --output result.jpg
python yolov8.py --config ../config/yolov8.yaml --use-camera --camera-id 0
```

**C++：** 在 `cv/build` 目录下：

```bash
./examples/yolov8 examples/yolov8/config/yolov8.yaml
./examples/yolov8 examples/yolov8/config/yolov8.yaml --image /path/to/image.jpg --output result.jpg
./examples/yolov8 examples/yolov8/config/yolov8.yaml --use-camera
```

启用 MPP 原生 NV12 DMA 输入：

```bash
./examples/yolov8 examples/yolov8/config/yolov8.yaml \
  --use-camera --use-mpp --mpp-vi
```

MPP 只负责采集并持有原生帧，`auto` 根据帧的 `NV12 + DMA-BUF`
属性选择 OpenCL。两者配置保持独立。用于显示和画框的 BGR 转换在
推理结束后执行，不作为模型预处理输入。C++ 示例会在首帧分别打印
MPP 帧输入类型和实际选择的 `opencl` / `cpu` 预处理后端。

OpenCL 和 MPP 都是可选构建项，开发板上启用两者的示例：

```bash
cmake -S . -B build \
  -DVISION_WITH_OPENCL=ON \
  -DVISION_WITH_MPP=ON \
  -DSTAGING_DIR=/path/to/output/staging
cmake --build build -j
```

只启用 OpenCL 并不会自动启用 MPP；只启用 MPP 也不会改变
`preprocess.backend` 的策略。

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/yolov8/yolov8n_no_dfl.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。
