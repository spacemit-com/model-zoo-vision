# YOLO-World 示例

开放词汇目标检测示例：用**文本提示词**（prompts）指定要检测的类别，模型无需重新训练即可检测任意词汇。

## 1. 模型与权重

- **模型类型**：开放词汇目标检测（YOLO-World）
- **模型文件**（与 `config/yolo_world.yaml` 一致）：
  - 检测：`~/.cache/models/vision/yolo_world/yolov8s-worldv2.dynq.onnx`
  - CLIP 文本编码：`~/.cache/models/vision/yolo_world/clip_text.onnx`
- **下载**：在仓库根目录执行：

```bash
bash examples/yolo_world/scripts/download_models.sh
```

会从 archive（`https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yolo_world/`）
  下载到 `~/.cache/models/vision/yolo_world/`（已存在则跳过）。

### 开放词汇的工作方式

YOLO-World 有**两个输入**：图像 + 文本嵌入。文本提示词经 CLIP 文本编码器 + BPE 分词器转成嵌入，
作为第二输入喂给检测模型。本示例的使用约定：

- 文本词汇（= 类别）由 `prompts` 指定：`config` 里给默认值，运行时可用 `--prompts` 覆盖。
- 文本嵌入**惰性缓存**：首次或词汇变更时才调用 CLIP 编码，重复同一词汇直接复用缓存，
  稳态推理是纯视觉、零文本开销。
- **BPE 合并表**（`assets/clip/bpe_merges.txt`）随仓库走，不需下载。

## 2. 配置文件说明（config/yolo_world.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | 检测 ONNX 模型路径 | `~/.cache/models/vision/yolo_world/yolov8s-worldv2.dynq.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/014_bus.jpg` |
| `default_params.clip_model_path` | CLIP 文本编码 ONNX | `~/.cache/models/vision/yolo_world/clip_text.onnx` |
| `default_params.bpe_merges_path` | BPE 合并表（仓库相对路径） | `assets/clip/bpe_merges.txt` |
| `default_params.prompts` | 默认文本词汇（= 类别） | `[person, bus, car]` |
| `default_params.conf_threshold` | 置信度阈值 | `0.25` |
| `default_params.iou_threshold` | NMS IoU 阈值 | `0.45` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `[SpaceMITExecutionProvider]` |

说明：`prompts` 支持列表（`[a, b, c]`）或逗号分隔字符串（`"a,b,c"`）。类别标签直接来自当前词汇，
无需 `label_file_path`（画框/打印时用运行时词汇）。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/yolo_world.py`）：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/yolo_world/config/yolo_world.yaml` |
| `--image` | 输入图片路径 | 使用 YAML 中 `test_image` |
| `--output` | 输出图片路径 | `yolo_world_result.jpg` |
| `--prompts` | 覆盖词汇，逗号分隔 `"person,bus,car"` | 未指定时使用配置文件默认值 |
| `--conf-threshold` / `--iou-threshold` | 阈值 | 来自配置文件|
| `--use-camera` / `--camera-id` | 摄像头输入 / 设备 ID | 关闭 / `0` |
| `--model-path` | 覆盖 YAML 中的 `model_path` | 无 |

**C++ 示例**：第一个参数是配置文件路径，可选
`--image`、`--output`、`--prompts`、`--model-path`。

> 运行时 `--prompts` 覆盖 YAML 默认词汇；留空则用配置文件的 `prompts`。

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

摄像头模式需要可用的摄像头设备和 OpenCV 图形显示环境；无桌面的 SSH 会话可先运行图片示例。

**Python：** 在仓库根目录运行：

```bash
python examples/yolo_world/python/yolo_world.py --config examples/yolo_world/config/yolo_world.yaml
python examples/yolo_world/python/yolo_world.py --config examples/yolo_world/config/yolo_world.yaml --prompts "person,bus,car" --output yolo_world_result.jpg
python examples/yolo_world/python/yolo_world.py --config examples/yolo_world/config/yolo_world.yaml --use-camera --prompts "person,dog"
```

**C++：** 在 `build` 目录运行：

```bash
./examples/yolo_world ../examples/yolo_world/config/yolo_world.yaml
./examples/yolo_world ../examples/yolo_world/config/yolo_world.yaml --prompts "person,bus,car" --output yolo_world_result.jpg
```

## 5. 故障排查

- **模型未找到**：确认已执行 `bash examples/yolo_world/scripts/download_models.sh`，检测/CLIP 两个模型都在
  `~/.cache/models/vision/yolo_world/` 下。
- **bpe_merges 打不开**：`bpe_merges_path` 走仓库相对路径 `assets/clip/bpe_merges.txt`，
  从仓库根目录运行、或用绝对路径。
- **检出框标签是数字而非词**：确认用的是带 prompts 的推理路径（C++ `VisionServiceRequest.prompts`、
  Python `infer_image(..., prompts=[...])`）；Python 侧需重新编译安装带 `infer_image_prompts` 的 wheel。
- **Python 传 prompts 报错**：pybind 扩展需包含 `infer_image_prompts` 绑定，改动后要重编扩展。

- **引擎提示 Einsum 不支持**：所用 SpaceMIT 引擎版本尚不支持该权重的 Einsum 表达式。
  可更新到支持该算子的引擎，或将 YAML 的 `default_params.providers` 改为
  `[CPUExecutionProvider]` 后运行；CPU 路径的延迟与硬件加速路径不同。
