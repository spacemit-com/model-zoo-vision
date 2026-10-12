# RT-DETR 示例

COCO 目标检测示例：对图像执行 RT-DETR 推理，输出检测框、置信度与类别标签，并保存可视化结果。

## 1. 模型与权重

- **模型类型**：COCO 目标检测（RT-DETR）
- **默认模型文件**：`~/.cache/models/vision/rtdetr/rtdetrv2_s.fp16.onnx`
- **可选模型文件**：`rtdetr_r18.fp16.onnx`
- **默认测试图片**：`~/.cache/assets/image/006_test.jpg`
- **类别标签文件**：`assets/labels/coco.txt`

在仓库根目录运行：

```bash
bash examples/rtdetr/scripts/download_models.sh
bash scripts/download_assets.sh
```

下载脚本从归档服务器的 `vision/rtdetr/` 下载 RT-DETR-R18 与 RT-DETRv2-S 两份模型并校验 SHA-256，已有缓存则跳过；可通过 `RTDETR_BASE_URL` 覆盖远端地址。

### 模型输入输出契约

两份模型的输入为 640×640 RGB 图像和原图尺寸。预处理将原图直接缩放，像素值除以 255；ONNX 图内已完成 TopK、得分计算和框解码，输出原图 `xyxy` 像素坐标，不再执行 NMS。统一检测接口中的 `iou_threshold` 参数不影响结果。

## 2. 配置文件说明（config/rtdetr.yaml）

`config/rtdetr.yaml` 的主要字段：

| 字段 | 作用 | 默认值 |
| --- | --- | --- |
| `model_path` | ONNX 权重路径 | `~/.cache/models/vision/rtdetr/rtdetrv2_s.fp16.onnx` |
| `test_image` | 输入图片 | `~/.cache/assets/image/006_test.jpg` |
| `label_file_path` | COCO 80 类名称 | `assets/labels/coco.txt` |
| `default_params.conf_threshold` | 输出分数阈值 | `0.3` |
| `default_params.num_threads` | 推理线程数 | `8` |
| `default_params.providers` | 推理 provider | `SpaceMITExecutionProvider` |
| `default_params.preprocess.backend` | 图像预处理后端 | `auto` |

默认配置使用 RT-DETRv2-S；两份模型共用 COCO 80 类标签。

## 3. 命令行参数

Python 示例支持 `--config`、`--model-path`、`--image`、`--output` 和 `--conf-threshold`。C++ 示例的第一个参数为配置文件路径，并支持 `--model-path`、`--image` 和 `--output`。

## 4. 运行示例

从仓库根目录运行 C++ 示例（先构建项目）：

```bash
./build/examples/rtdetr examples/rtdetr/config/rtdetr.yaml
./build/examples/rtdetr examples/rtdetr/config/rtdetr.yaml \
  --image /path/to/image.jpg --output rtdetrv2_result.jpg
```

使用与当前源码对应的 `spacemit_vision` Python 包：

```bash
python3 examples/rtdetr/python/rtdetr.py
python3 examples/rtdetr/python/rtdetr.py \
  --image /path/to/image.jpg --output rtdetrv2_result.jpg
```

从源码树运行 Python 示例时，先构建原生绑定并更新源码包中的运行库，避免加载旧版本 `.so`：

```bash
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build --target _vision_service_cpp -j8
(cd src/python && PYTHON_BIN=python3 ./build_wheel.sh ../../build)
PYTHONPATH=src/python python3 examples/rtdetr/python/rtdetr.py
```

## 5. 故障排查

- 模型找不到：先运行模型脚本，或通过 `--model-path` 指定现有权重。
- 测试图找不到：运行 `bash scripts/download_assets.sh`，或通过 `--image` 指定图片。
- 类别名称不匹配：确认配置使用紧凑的 80 类 `assets/labels/coco.txt`，不要使用 RF-DETR 的稀疏 COCO 标签文件。
