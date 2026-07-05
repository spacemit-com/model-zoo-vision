# YOLOE 示例

开放词汇**实例分割**示例：用文本提示词指定类别，模型对每个实例输出检测框 + 类别 + **分割 mask**。

## 1. 模型与权重

- **模型类型**：开放词汇实例分割（YOLOE）
- **模型文件**（与 `config/yoloe.yaml` 一致）：
  - 分割：`~/.cache/models/vision/yoloe/yoloe-v8s-seg.dynq.onnx`
  - MobileCLIP 文本编码：`~/.cache/models/vision/yoloe/mobileclip.q.onnx`
- **下载**：在本示例目录下执行
  `bash scripts/download_models.sh`
  会从 archive（`https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yoloe/`）
  下载到 `~/.cache/models/vision/yoloe/`（已存在则跳过）。

### 开放词汇 + 分割的工作方式

YOLOE 有**两个输入**：图像 + 文本嵌入；**两个输出**：检测（含 mask 系数）+ proto。
文本提示词经 MobileCLIP + BPE 分词器转成嵌入作为第二输入；mask 由 proto 与每个实例的
mask 系数相乘、阈值化后 resize 回原图得到。处理约定：

- 文本词汇（= 类别）由 `prompts` 指定：`config` 给默认值，运行时 `--prompts` 覆盖。
- 文本嵌入**惰性缓存**：词汇不变则复用，稳态零文本开销。
- **分割/检测自动判定**：模型有 ≥2 个输出时按分割处理（出 mask）；单输出则退化为纯检测（mask 为空）。
- **BPE 合并表**（`assets/clip/bpe_merges.txt`）与 YOLO-World 共用，随仓库走，不需下载。
- 画图时不同类别的 mask 自动用不同颜色（公共 `draw_segmentation` 的固定调色板，按类别索引取色）。

## 2. 配置文件说明（config/yoloe.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | 分割 ONNX 模型路径 | `~/.cache/models/vision/yoloe/yoloe-v8s-seg.dynq.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/014_bus.jpg` |
| `default_params.clip_model_path` | MobileCLIP 文本编码 ONNX | `~/.cache/models/vision/yoloe/mobileclip.q.onnx` |
| `default_params.bpe_merges_path` | BPE 合并表（仓库相对路径） | `assets/clip/bpe_merges.txt` |
| `default_params.prompts` | 默认文本词汇（= 类别） | `[person, bus]` |
| `default_params.conf_threshold` | 置信度阈值 | `0.25` |
| `default_params.iou_threshold` | NMS IoU 阈值 | `0.45` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

说明：`prompts` 支持列表或逗号分隔字符串。类别标签来自当前词汇，无需 `label_file_path`。

## 3. 命令行参数

**Python 示例**（`python/yoloe.py`）：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/yoloe/config/yoloe.yaml` |
| `--image` | 输入图片路径 | 使用 yaml 中 `test_image` |
| `--output` | 输出图片路径 | `yoloe_result.jpg` |
| `--prompts` | 覆盖词汇，逗号分隔 `"person,bus"` | 空 → 用 config 默认 |
| `--conf-threshold` / `--iou-threshold` | 阈值 | 来自 config |
| `--use-camera` / `--camera-id` | 摄像头输入 / 设备 ID | 关 / `0` |
| `--model-path` | 覆盖 yaml 中的 `model_path` | 无 |

**C++ 示例**（需在 `cv/build` 下先编译）：第一个参数是配置文件路径，可选
`--image`、`--output`、`--prompts`、`--model-path`。

## 4. 运行示例

**Python：**

```bash
cd examples/yoloe/python
python yoloe.py --config ../config/yoloe.yaml
python yoloe.py --config ../config/yoloe.yaml --prompts "person,bus" --output yoloe_result.jpg
python yoloe.py --config ../config/yoloe.yaml --use-camera --prompts "person,dog"
```

**C++：** 在 `cv/build` 目录下：

```bash
./examples/yoloe examples/yoloe/config/yoloe.yaml
./examples/yoloe examples/yoloe/config/yoloe.yaml --prompts "person,bus" --output yoloe_result.jpg
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`，分割/MobileCLIP 两个模型都在
  `~/.cache/models/vision/yoloe/` 下。
- **bpe_merges 打不开**：`bpe_merges_path` 走仓库相对路径 `assets/clip/bpe_merges.txt`，
  从仓库根目录运行、或改绝对路径。
- **没有 mask 只有框**：模型需为分割导出（≥2 个输出）；单输出模型会退化为纯检测。
- **框标签是数字而非词**：确认走带 prompts 的推理路径；Python 侧需重编含 `infer_image_prompts`
  绑定的 wheel。
