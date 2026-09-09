# PP-LiteSeg 示例

语义分割示例：对图像进行 Cityscapes 19 类分割，输出按类别着色的掩膜叠加图。

## 1. 模型与权重

- **模型类型**：语义分割（PP-LiteSeg）
- **默认模型文件**：`~/.cache/models/vision/pp_liteseg/pp_liteseg.q.onnx`（与 `config/pp_liteseg.yaml` 中的 `model_path` 一致）
- **下载**：在本示例目录下执行 `bash scripts/download_models.sh`，模型会下载到上述缓存路径。

**数据（测试图片）**：默认 `test_image` 指向
`~/.cache/assets/image/009_test_unet.jpg`。若尚未下载资源，请在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

### 模型格式说明

输入采用 NCHW 布局，尺寸从 ONNX 读取。预处理将 BGR 转为 RGB，等比缩放后
放在输入画布左上角，并使用 `(pixel / 255 - 0.5) / 0.5` 归一化。
后处理裁出有效区域，再用最近邻插值还原原图尺寸。

当前实现要求模型输出为 int32，支持二维标签图或带批次/类别维的输出；
有类别维时执行 argmax。不支持直接替换成 float32 输出的导出模型。

返回结果按类别拆成原图大小的二值掩膜，当前实现只返回出现的类别1至18，
不返回类别0；`score` 固定为1，不表示预测置信度。

## 2. 配置文件说明（config/pp_liteseg.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/pp_liteseg/pp_liteseg.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/009_test_unet.jpg` |
| `default_params.num_threads` | 推理线程数 | `8` |
| `default_params.num_classes` | 分割类别数 | `19` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

通用字段（如 `class`、`default_params` 结构）与其他示例一致。
CPU 对照可将 provider 改为 `CPUExecutionProvider`。
若模型输出为四维且类别维已确定，实现会从输出形状更新类别数。
本模型不执行 NMS，通用 API 的置信度和 IoU 阈值参数不影响分割结果。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/pp_liteseg.py`）：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | 示例内的 `config/pp_liteseg.yaml` |
| `--image` | 输入图片路径 | 使用 YAML 中的 `test_image` |
| `--output` | 输出图片路径 | `pp_liteseg_result.jpg` |
| `--model-path` | 覆盖 YAML 中的 `model_path` | 无 |
| `--alpha` | Python 备用绘图路径的掩膜透明度 | `0.4` |

当前 PP-LiteSeg 支持公共 `Draw`，Python 默认调用该绘图接口，因此
`--alpha` 不影响正常运行时的输出。Python 的相对 `--image` 路径按仓库
根目录解析，建议自定义图片时使用绝对路径。

**C++ 示例**：第一个参数为配置文件路径，可选 `--image`、`--output`、
`--model-path` 和 `--help`。默认图片和输出文件名与 Python 一致。

## 4. 运行示例

**Python：** 需安装 `spacemit_vision` 包，以及 OpenCV、NumPy 和 PyYAML。
从仓库根目录进入示例目录：

```bash
cd examples/pp_liteseg/python
python3 pp_liteseg.py --config ../config/pp_liteseg.yaml
python3 pp_liteseg.py --config ../config/pp_liteseg.yaml \
  --image /path/to/image.jpg --output result.jpg
```

**C++：** 按仓库 README 构建并启用 examples 后，在仓库的 `build/` 目录下：

```bash
./examples/pp_liteseg ../examples/pp_liteseg/config/pp_liteseg.yaml
./examples/pp_liteseg ../examples/pp_liteseg/config/pp_liteseg.yaml \
  --image /path/to/image.jpg --output result.jpg
```

只传配置即可使用默认测试图片，结果保存到当前工作目录。
两种示例均通过公共服务完成推理和掩膜绘制。

## 5. 故障排查

- **模型未找到**：在本示例目录执行 `bash scripts/download_models.sh`，确认 `model_path` 指向下载后的模型。
- **测试图片未找到**：在仓库根目录执行 `bash scripts/download_assets.sh`，或通过 `--image` 指定图片。
- **Python 包或扩展导入失败**：确认已安装当前构建的 `spacemit_vision` 包，且运行库依赖可用。
- **提示 expected int32 output tensor**：所用模型输出类型不符合当前实现，检查是否替换了不同导出格式的权重。
- **没有掩膜输出**：当前实现跳过类别0；若图像只预测到类别0，将返回空结果，C++ 示例不会保存结果图。
