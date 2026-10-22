# YOLO26n-Sem 示例

语义分割示例：使用 YOLO26n-Sem 对图像执行 Cityscapes 19 类逐像素
分类，并保存叠加语义掩码后的结果图。`-sem` 表示语义分割，不同于
返回独立目标掩码和检测框的 `-seg` 实例分割模型。

## 1. 模型与权重

- **模型类型**：YOLO26n 语义分割
- **训练数据集**：Cityscapes，19 类
- **固定输入**：RGB `float32 [1, 3, 512, 1024]`
- **模型输出**：logits `float32 [1, 19, 512, 1024]`
- **默认模型文件**：`~/.cache/models/vision/yolo26_sem/yolo26n-sem.fp16.onnx`
- **同系列**：`yolo26s-sem.fp16.onnx`、`yolo26m-sem.fp16.onnx`（用 `--model-path` 切换）

在仓库根目录执行：

```bash
bash examples/yolo26_sem/scripts/download_models.sh
```

脚本从 SpaceMIT 模型归档的 `vision/yolo26_sem/` 下载 n/s/m 三个 FP16
权重到 `~/.cache/models/vision/yolo26_sem/`。

默认测试图片为 `~/.cache/assets/image/009_test_unet.jpg`。若图片尚未
准备，在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/yolo26_sem.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/yolo26_sem/yolo26n-sem.fp16.onnx` |
| `test_image` | 默认测试图片 | `~/.cache/assets/image/009_test_unet.jpg` |
| `label_file_path` | Cityscapes 19 类标签 | `assets/labels/cityscapes.txt` |
| `class` | 工厂注册类 | `deploy.yolo26_sem.YOLO26SemanticSegmentor` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `8` |
| `default_params.num_classes` | 输出类别数 | `19` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `[SpaceMITExecutionProvider]` |
| `default_params.preprocess.backend` | 图像预处理后端 | `auto` |

预处理遵循 Ultralytics 推理契约：图像按比例缩放并居中填充至
`1024x512`，填充值为 `114`，然后 BGR 转 RGB 并除以 `255`。后处理
先移除 letterbox 填充，再将各类别 logits 双线性还原至原图尺寸，最后
执行逐像素 `argmax`。模型不输出实例、检测框或逐实例置信度，因此
`conf_threshold` 和 `iou_threshold` 不适用。

## 3. 命令行 / API 参数（与本示例相关）

Python 示例支持：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/yolo26_sem/config/yolo26_sem.yaml` |
| `--image` | 覆盖输入图片 | YAML 中的 `test_image` |
| `--output` | 输出可视化图片 | `yolo26_sem_result.jpg` |
| `--model-path` | 覆盖 YAML 中的模型路径 | 无 |

C++ 示例的第一个参数必须是配置文件路径，并支持相同的 `--image`、
`--output` 和 `--model-path` 覆盖参数。

模型通过统一 `VisionService` 创建，支持 `ImageInput`、分割意图和
`Draw()`。返回值是图像中实际出现类别的二值掩码，每张掩码都与原图
尺寸一致；同类像素合并为一张语义掩码。

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

**Python：** 在仓库根目录运行：

```bash
python examples/yolo26_sem/python/yolo26_sem.py --config examples/yolo26_sem/config/yolo26_sem.yaml
python examples/yolo26_sem/python/yolo26_sem.py --config examples/yolo26_sem/config/yolo26_sem.yaml --image /path/to/image.jpg --output result.jpg
```

**C++：** 在 `build` 目录运行：

```bash
./examples/yolo26_sem ../examples/yolo26_sem/config/yolo26_sem.yaml
./examples/yolo26_sem ../examples/yolo26_sem/config/yolo26_sem.yaml --image /path/to/image.jpg --output result.jpg
```

## 5. 故障排查

- **模型未找到**：执行
  `bash examples/yolo26_sem/scripts/download_models.sh`，或用
  `--model-path` 指定同一导出契约的模型。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或传入
  `--image`。
- **模型形状或类型错误**：本实现只接受 `float32 [1,3,H,W]` 输入和
  空间尺寸一致的 `float32 [1,19,H,W]` logits 输出。
- **颜色或掩码位置异常**：确认输入是 OpenCV BGR 三通道图片；代码会
  完成 RGB 转换、letterbox 及反向几何恢复。
- **Python 无法导入原生扩展**：构建并安装当前源码对应的 `spacemit_vision` 包，确认平台和 Python 版本匹配。
