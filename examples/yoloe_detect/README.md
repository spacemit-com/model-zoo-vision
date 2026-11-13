# YOLOE Detect 示例

开放词汇目标检测示例，支持文本提示、视觉提示、文本与视觉融合三种模式，输出检测框、类别和置信度。
原有 `examples/yoloe` 保留为文本提示实例分割示例；本示例只做检测，不输出 mask。

## 1. 模型与权重

模型缓存目录为 `~/.cache/models/vision/yoloe_detect/`，使用同一套配套导出的三个模型：

| 模型文件 | 用途 |
|----------|------|
| `yoloe26n_vp_det_sim.fp16.onnx` | 检测器，三种模式共用 |
| `yoloe26n_text_sim.fp16.onnx` | 文本提示编码，供 text / fused 使用 |
| `yoloe26n_vpe_sim.fp16.onnx` | 视觉提示编码，供 visual / fused 使用 |

在仓库根目录下载模型和测试图片：

```bash
bash examples/yoloe_detect/scripts/download_models.sh
bash scripts/download_assets.sh
```

下载脚本从 `https://archive.spacemit.com/spacemit-ai/model_zoo/vision/yoloe_detect/`
获取上述三个文件，已存在的非空文件会跳过。可用环境变量 `YOLOE_DETECT_BASE_URL`
覆盖下载地址。BPE 合并表复用仓库的 `assets/clip/bpe_merges.txt`，不需单独下载。

默认目标图和参考图均为 `~/.cache/assets/image/014_bus.jpg`。
三个模型必须使用配套导出版本，不能与原 YOLOE 分割示例的权重、文本编码器混用。
这里的“三种”指提示模式，并非检测器具有三个 ONNX 输入。

## 2. 配置文件说明（config/yoloe_detect.yaml）

三种模式使用同一份模型配置。

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | 检测器路径 | 缓存中的 `yoloe26n_vp_det_sim.fp16.onnx` |
| `test_image` | 默认目标图片 | `~/.cache/assets/image/014_bus.jpg` |
| `default_params.text_model_path` | 文本编码器路径 | 缓存中的 `yoloe26n_text_sim.fp16.onnx` |
| `default_params.vpe_model_path` | 视觉编码器路径 | 缓存中的 `yoloe26n_vpe_sim.fp16.onnx` |
| `default_params.bpe_merges_path` | BPE 合并表 | `assets/clip/bpe_merges.txt` |
| `default_params.prompts` | 默认文本类别列表 | `[person, bus]` |
| `default_params.conf_threshold` | 置信度阈值 | `0.2` |
| `default_params.num_threads` | 检测器推理线程 / AI core 数 | `8` |
| `default_params.prompt_num_threads` | CPU 提示编码线程数 | `2` |
| `default_params.providers` | 检测器执行提供方，单元素列表 | `[SpaceMITExecutionProvider]` |
| `default_params.preprocess.backend` | 图像预处理后端 | `cpu` |

文本与视觉编码器按需加载并使用 CPU；检测器可切换为 `[CPUExecutionProvider]`。
当前接入使用 BGR8 图片，保持宽高比 letterbox 到 640×640，转 RGB 并除以 255。
类别名来自本次提示，无需 COCO 标签文件，支持 1～80 个不同类别。
检测后处理使用 Top-K，不执行 NMS；公共 API 的 `iou_threshold` 会校验，但不参与后处理。
`conf_threshold` 的正数覆盖配置，`<=0` 使用默认值；`max_det=0` 返回空结果，
正数进一步限制数量，模型最多输出 300 个候选。

### 视觉提示文件（visual_prompts.yaml）

这是参考图片与框的数据文件，不是第二份模型配置。默认包含公交车图片上的 person / bus 参考框。
更换目标图用 `--image`；更换参考图则需同时修改参考图片路径和对应的框：

```yaml
- class_name: person
  image_path: ~/.cache/assets/image/014_bus.jpg
  bboxes:
    - [221.52, 405.80, 344.98, 857.54]
    - [50.00, 399.00, 240.00, 900.00]
- class_name: bus
  image_path: ~/.cache/assets/image/014_bus.jpg
  bboxes:
    - [17.00, 231.00, 801.00, 768.00]
```

坐标为参考原图中的 `[x1, y1, x2, y2]`，必须在图内且框有正面积；框过小、在视觉编码网格上为空时会报错。
同一条记录的多个框取并集；同类可写多条记录引用不同图片，编码后聚合为一个类别。
`image_path` 支持绝对路径、`~/`，相对路径以提示文件所在目录为基准。

- `text`：仅使用文本类别，未传 `--prompts` 时用配置中的 `prompts`。
- `visual`：仅使用参考图和框，类别顺序由提示文件中 `class_name` 首次出现的顺序决定，不使用默认文本提示。
- `fused`：融合文本和视觉特征。文本类别由 `--prompts` 或配置给出，每个文本类别必须有同名视觉参考，不允许多出未声明的视觉类别。

提示不变时复用编码结果；更换类别、参考图片内容或框后会重新编码。

## 3. 命令行 / API 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | `text` / `visual` / `fused` | `text` |
| `--image` | 目标图片 | YAML 的 `test_image` |
| `--prompts` | 逗号分隔文本类别，visual 模式不使用此参数 | YAML 的 `prompts` |
| `--visual-prompts` | 视觉提示文件，visual / fused 使用 | 模型配置目录上一级的 `visual_prompts.yaml` |
| `--model-path` | 覆盖检测器权重路径，不替换编码器 | 无 |
| `--conf-threshold` | 覆盖置信度阈值 | YAML 的 `conf_threshold` |
| `--output` | 结果图片 | `yoloe_detect_result.jpg` |

C++ 第一个参数为配置路径；Python 使用 `--config`，省略时读取本示例的默认配置。
示例暂不提供摄像头输入。

Python API 的视觉提示格式如下；仅传 `visual_prompts` 为视觉模式，同时传非空 `prompts` 为融合模式：

```python
references = [{
    "class_name": "person",
    "image": reference_bgr,  # uint8 H×W×3 numpy 数组
    "bboxes": [[50, 399, 240, 900]],
}]
status, results = service.infer_image(image_bgr, visual_prompts=references)
status, results = service.infer_image(
    image_bgr, prompts=["person"], visual_prompts=references,
)
```

## 4. 运行示例

运行前按仓库 README 编译 C++ 示例。Python 示例需要与当前平台及 Python 版本匹配、
包含视觉提示接口的 `spacemit_vision` 包，以及 `numpy`、`opencv-python`（或系统 OpenCV Python 包）、`PyYAML`。
更新 C++ 库后也需重编 Python 原生扩展。

Python：在仓库根目录运行，三条命令分别对应三种提示模式：

```bash
python3 examples/yoloe_detect/python/yoloe_detect.py --mode text --prompts "person,bus" --output text_result.jpg
python3 examples/yoloe_detect/python/yoloe_detect.py --mode visual --output visual_result.jpg
python3 examples/yoloe_detect/python/yoloe_detect.py --mode fused --prompts "person,bus" --output fused_result.jpg
```

自定义目标图与视觉参考：

```bash
python3 examples/yoloe_detect/python/yoloe_detect.py \
  --config examples/yoloe_detect/config/yoloe_detect.yaml \
  --mode visual --image /path/to/target.jpg \
  --visual-prompts /path/to/visual_prompts.yaml --output result.jpg
```

C++：在 `build` 目录运行：

```bash
./examples/yoloe_detect ../examples/yoloe_detect/config/yoloe_detect.yaml --mode text --prompts "person,bus" --output text_result.jpg
./examples/yoloe_detect ../examples/yoloe_detect/config/yoloe_detect.yaml --mode visual --output visual_result.jpg
./examples/yoloe_detect ../examples/yoloe_detect/config/yoloe_detect.yaml --mode fused --prompts "person,bus" --output fused_result.jpg
```

自定义目标图与视觉参考：

```bash
./examples/yoloe_detect ../examples/yoloe_detect/config/yoloe_detect.yaml \
  --mode visual --image /path/to/target.jpg \
  --visual-prompts /path/to/visual_prompts.yaml --output result.jpg
```

## 5. 故障排查

- 模型缺失：检查下载脚本与配置中的三个权重路径，确认远端已有对应文件；不要拿旧分割权重替代。
- 提示类别不匹配：fused 模式的文本类别和视觉类别须逐项对应；visual 模式不要传 `--prompts`。
- 参考框无效：坐标必须对应参考原图，而非目标图或缩放后的图片，并避免过小的框。
- Python 缺少 `infer_image_visual_prompts`：当前加载的是旧原生扩展，需要重新构建并安装当前版本。
