# RF-DETR 示例

COCO 目标检测示例：对图像执行 RF-DETR 推理，输出检测框、置信度与类别标签，
并保存可视化结果。

## 1. 模型与权重

- **模型类型**：目标检测（RF-DETR）
- **默认模型文件**：`~/.cache/models/vision/rfdetr/rfdetr-nano.fp16.onnx`
- **可选模型文件**：`rfdetr-small.fp16.onnx`、`rfdetr-medium.fp16.onnx`
- **默认测试图片**：`~/.cache/assets/image/006_test.jpg`
- **类别标签文件**：`assets/labels/coco_sparse.txt`

在组件根目录执行：

```bash
bash examples/rfdetr/scripts/download_models.sh
bash scripts/download_assets.sh
```

下载脚本会准备 Nano、Small、Medium 三套模型。本地仓库根目录存在同名模型时
优先复制，否则从归档服务器的 `vision/rfdetr/` 下载。

### 模型输入输出契约

三套模型的输入分辨率分别为 384×384、512×512 和 576×576，输入输出名称及
语义相同：

| 名称 | 类型与形状 | 说明 |
|------|------------|------|
| `input` | float32 `[1, 3, H, W]` | RGB，ImageNet mean/std 归一化 |
| `dets` | float32 `[1, 300, 4]` | 归一化的 `cxcywh` 检测框 |
| `labels` | float32 `[1, 300, 91]` | 91 个稀疏 COCO 类别槽位的 logits |

预处理将原图直接双线性缩放到模型输入尺寸。后处理对类别 logits 执行 sigmoid，
在 query×class 中取置信度最高的结果，再将检测框还原到原图坐标。RF-DETR 不
执行 NMS，因此统一检测接口中的 `iou_threshold` 不生效。

RF-DETR 的 COCO 输出使用原始 category id 作为类别槽位，包含未使用的索引，
不能换成紧凑的 80 类 `coco.txt`；默认配置使用对应的 91 行标签文件。

## 2. 配置文件说明（config/rfdetr.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/rfdetr/rfdetr-nano.fp16.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/006_test.jpg` |
| `label_file_path` | 稀疏 COCO 类别标签文件 | `assets/labels/coco_sparse.txt` |
| `default_params.conf_threshold` | 置信度阈值 | `0.3` |
| `default_params.max_det` | 最多保留的检测结果数 | `300` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `8` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |
| `default_params.preprocess.backend` | 预处理策略：`cpu` / `auto` / `opencl` | `auto` |

默认配置使用 Nano。切换到 Small 或 Medium 时只需通过 `--model-path` 指定模型，
检测器会从 ONNX 输入自动读取分辨率。

## 3. 命令行参数（与本模块相关）

Python 示例支持：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/rfdetr/config/rfdetr.yaml` |
| `--image` | 覆盖输入图片 | yaml 中的 `test_image` |
| `--output` | 输出可视化图片 | `rfdetr_result.jpg` |
| `--model-path` | 覆盖 yaml 中的模型路径 | 无 |
| `--conf-threshold` | 覆盖置信度阈值 | 使用 yaml 配置 |

C++ 示例的第一个参数必须是配置文件路径，并支持 `--image`、`--output` 和
`--model-path`。

## 4. 运行示例

在组件根目录运行 C++ 示例：

```bash
./build/examples/rfdetr examples/rfdetr/config/rfdetr.yaml
./build/examples/rfdetr examples/rfdetr/config/rfdetr.yaml \
  --model-path ~/.cache/models/vision/rfdetr/rfdetr-small.fp16.onnx
./build/examples/rfdetr examples/rfdetr/config/rfdetr.yaml \
  --model-path ~/.cache/models/vision/rfdetr/rfdetr-medium.fp16.onnx
./build/examples/rfdetr examples/rfdetr/config/rfdetr.yaml \
  --image /path/to/image.jpg --output rfdetr_result.jpg
```

使用已安装且由当前源码构建的 `spacemit_vision` Python 包：

```bash
python3 examples/rfdetr/python/rfdetr.py
python3 examples/rfdetr/python/rfdetr.py \
  --model-path ~/.cache/models/vision/rfdetr/rfdetr-small.fp16.onnx
```

直接使用当前源码树时，先构建原生扩展并更新源码包中的运行库：

```bash
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build -j8 --target _vision_service_cpp
(cd src/python && PYTHON_BIN=python3 ./build_wheel.sh ../../build)
PYTHONPATH=src/python \
  python3 examples/rfdetr/python/rfdetr.py
```

## 5. 故障排查

- **模型未找到**：执行 `bash examples/rfdetr/scripts/download_models.sh`，
  或使用 `--model-path` 指定模型。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或使用
  `--image` 指定图片。
- **类别名称错位**：确认配置使用 `assets/labels/coco_sparse.txt`，不要改为
  紧凑的 80 类 COCO 标签文件。
- **Python 无法导入原生扩展**：先构建 `_vision_service_cpp`，并使用当前构建
  生成的 Python 包。
