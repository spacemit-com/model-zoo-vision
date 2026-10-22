# YOLO26-Depth 示例

单目深度估计示例：从一张普通 RGB 图像预测逐像素距离，并保存深度
伪彩色图。模型返回的是 metric depth（单位为米），不是双目视差。

## 1. 模型与权重

- **模型类型**：YOLO26 n/s/m 单目深度估计
- **固定输入**：RGB `float32 [1, 3, 768, 768]`
- **模型输出**：深度 `float32 [1, 1, 768, 768]`，单位为米
- **默认模型文件**：`~/.cache/models/vision/yolo26_depth/yolo26n-depth.fp16.onnx`

| 规格 | 模型文件 | 用法 |
|------|----------|------|
| n（默认） | `yolo26n-depth.fp16.onnx` | 直接使用默认配置 |
| s | `yolo26s-depth.fp16.onnx` | 通过 `--model-path` 切换 |
| m | `yolo26m-depth.fp16.onnx` | 通过 `--model-path` 切换 |

在仓库根目录下载模型：

```bash
bash examples/yolo26_depth/scripts/download_models.sh
```

脚本会从 SpaceMIT 模型归档的 `vision/yolo26_depth/` 下载 n/s/m 三个
FP16 权重到 `~/.cache/models/vision/yolo26_depth/`。三个权重具有相同
的输入输出契约，因此共用一个配置和推理实现。

默认测试图片为 `~/.cache/assets/image/006_test.jpg`。若图片尚未准备，
在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/yolo26_depth.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/yolo26_depth/yolo26n-depth.fp16.onnx` |
| `test_image` | 默认测试图片 | `~/.cache/assets/image/006_test.jpg` |
| `class` | 工厂注册类 | `deploy.yolo26_depth.YOLO26DepthEstimator` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `8` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `[SpaceMITExecutionProvider]` |
| `default_params.preprocess.backend` | 图像预处理后端 | `auto` |

预处理按比例缩放并居中填充至 `768x768`，填充值为 `114`，然后
BGR 转 RGB 并除以 `255`。后处理移除 letterbox 填充并将深度图双线性
还原至原图尺寸。无效的非正值不会参与统计或着色；可视化采用近处暖色、
远处冷色。

## 3. 命令行 / API 参数（与本示例相关）

Python 示例支持：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/yolo26_depth/config/yolo26_depth.yaml` |
| `--image` | 覆盖输入图片 | YAML 中的 `test_image` |
| `--output` | 输出伪彩色图 | `yolo26_depth_result.jpg` |
| `--model-path` | 覆盖 YAML 中的模型路径 | 无 |

C++ 示例的第一个参数必须是配置文件路径，并支持相同的 `--image`、
`--output` 和 `--model-path` 参数。C++ 结果类型为 `vision::DepthMap`；
Python 可通过 `VisionServiceNative.infer_depth()` 获得 `float32 HxW`
NumPy 数组。

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

**Python：** 在仓库根目录运行：

```bash
python examples/yolo26_depth/python/yolo26_depth.py --config examples/yolo26_depth/config/yolo26_depth.yaml
python examples/yolo26_depth/python/yolo26_depth.py --config examples/yolo26_depth/config/yolo26_depth.yaml --image /path/to/image.jpg --output result.jpg
```

**C++：** 在 `build` 目录运行：

```bash
./examples/yolo26_depth ../examples/yolo26_depth/config/yolo26_depth.yaml
./examples/yolo26_depth ../examples/yolo26_depth/config/yolo26_depth.yaml --image /path/to/image.jpg --output result.jpg
```

深度结果的单位为米；可视化颜色只用于显示，不代表另一个深度单位。

## 5. 故障排查

- **模型未找到**：执行下载脚本，或用 `--model-path` 指定具有相同
  `[1,3,768,768] -> [1,1,768,768]` 契约的模型。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或传入
  `--image`。
- **深度图全黑或无有效值**：确认所用权重输出的是正数 metric depth，
  而不是尚未解码的 log-depth。
- **Python 无法导入原生扩展**：构建并安装当前源码对应的 `spacemit_vision` 包，确认平台和 Python 版本匹配。
