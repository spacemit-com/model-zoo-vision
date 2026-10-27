# YOLO26-Seg 实例分割示例

YOLO26-Seg 对 COCO 80 类目标输出检测框、类别、分数和逐实例掩码。这里的 `-seg` 是实例分割，不同于 `yolo26_sem` 的逐像素语义分类。

## 1. 模型与权重

- 默认权重：`yolo26n-seg.fp16.onnx`；同一配置可用 `--model-path` 切换 `yolo26s-seg.fp16.onnx` 或 `yolo26m-seg.fp16.onnx`。
- 输入：RGB `float32 [1,3,640,640]`；OpenCV BGR 图像由示例内部预处理。
- 输出：检测 `float32 [1,300,38]`（框、分数、类别和 32 个掩码系数）以及原型 `float32 [1,32,160,160]`。
- 数据集及标签：COCO 80 类，`assets/labels/coco.txt`。

在仓库根目录准备模型和默认测试图片：

```bash
bash examples/yolo26_seg/scripts/download_models.sh
bash scripts/download_assets.sh
```

模型脚本优先从仓库根目录复制这三个 ONNX 文件，校验 SHA-256 后放到 `~/.cache/models/vision/yolo26_seg/`。若根目录没有模型，脚本改从 `vision/yolo26_seg/` 归档目录下载；可通过 `YOLO26_SEG_BASE_URL` 覆盖地址。

## 2. 配置文件说明

唯一配置文件是 `config/yolo26_seg.yaml`。默认使用 n 模型、`~/.cache/assets/image/006_test.jpg`、8 个 CPU 线程和 `SpaceMITExecutionProvider`；`default_params.conf_threshold` 为 `0.25`，`mask_threshold` 为 `0.5`，`preprocess.backend` 为 `auto`。可通过 `--model-path` 和 `--image` 覆盖模型与图片。

预处理按比例缩放并居中填充到 640×640（填充值 114），再做 BGR→RGB、除以 255。模型输出为端到端候选，示例只按置信度过滤，不做额外 NMS；掩码由系数与原型相乘，还原到原图尺寸并按各自检测框裁剪。`iou_threshold` 因此不生效。返回的每个实例掩码都是与输入图同尺寸的 `uint8` 二值图。

## 3. 运行示例

先按仓库 README 构建 C++ 示例；运行 Python 示例前，安装与当前源码及平台匹配的 `spacemit_vision` 包。

在 `build` 目录运行 C++：

```bash
./examples/yolo26_seg ../examples/yolo26_seg/config/yolo26_seg.yaml
./examples/yolo26_seg ../examples/yolo26_seg/config/yolo26_seg.yaml --model-path "$HOME/.cache/models/vision/yolo26_seg/yolo26s-seg.fp16.onnx" --image "$HOME/.cache/assets/image/006_test.jpg" --output yolo26s_seg_result.jpg
```

在仓库根目录运行 Python：

```bash
python examples/yolo26_seg/python/yolo26_seg.py
python examples/yolo26_seg/python/yolo26_seg.py --model-path "$HOME/.cache/models/vision/yolo26_seg/yolo26m-seg.fp16.onnx" --image "$HOME/.cache/assets/image/006_test.jpg" --output yolo26m_seg_result.jpg
```

C++ 示例与 YOLOv8-Seg 一样打印检测总数；Python 示例逐项打印类别、分数、框坐标和掩码尺寸。两者都通过公共 `Draw()` API 保存可视化图。Python 还支持 `--conf-threshold`；两种示例均支持 `--output`。

## 4. 故障排查

- 模型不存在或校验失败：确认根目录 ONNX 文件完整，再运行下载脚本；归档尚未上传时不要依赖远端下载。
- 图像不存在：运行 `bash scripts/download_assets.sh` 或指定 `--image`。
- 提示输出契约不匹配：本示例仅支持上述两路端到端分割输出，不能直接换用原始 YOLO 分割头导出。
- Python 找不到扩展：构建并安装与当前源码版本相同的 `spacemit_vision` 包。
