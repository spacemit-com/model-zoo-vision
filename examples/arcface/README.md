# ArcFace 示例

人脸识别示例：对两张人脸图像提取特征并计算相似度，用于判断是否为同一人。与 YOLOv5-Face（仅检测人脸框）不同，本示例输出相似度分数。

## 1. 模型与权重

- **模型类型**：人脸识别（特征提取 + 相似度）
- **默认模型文件**：`~/.cache/models/vision/arcface/arcface_mobilefacenet_cut.q.onnx`（与 `config/arcface.yaml` 中 `model_path` 一致）
- **下载**：在仓库根目录执行：

```bash
bash examples/arcface/scripts/download_models.sh
```

会将模型下载到上述缓存路径。

**数据（测试图片）**：默认使用的两张人脸图 `test_image1`、`test_image2` 指向 `~/.cache/assets/image/003_face0.png`、`004_face1.png`。若尚未下载资源，请在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/arcface.yaml）

本示例使用的配置文件为 `config/arcface.yaml`。与**本示例强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/arcface/arcface_mobilefacenet_cut.q.onnx` |
| ONNX 输入尺寸 | 由模型自动读取，不能通过 YAML 改写 | 见模型文件 |
| `test_image1` | 示例用第一张人脸图像路径（相对仓库根目录或绝对） | `~/.cache/assets/image/003_face0.png` |
| `test_image2` | 示例用第二张人脸图像路径 | `~/.cache/assets/image/004_face1.png` |
| `default_params.num_threads` | 推理线程数 | `8` |
| `default_params.providers` | ONNX Runtime 执行提供方（单元素列表） | `[SpaceMITExecutionProvider]` |

说明：通用字段（如 `class`、`default_params` 结构）与其他示例一致，不在此重复。

`default_params.norm_std` 是归一化除数，默认为 `127.5`，计算为
`(RGB - 127.5) / norm_std`。该值必须是有限正数，且不能小到导致倒数溢出；
分发权重使用默认值即可。

### 输入要求与使用注意

- **输入要求**：模型输入为 **112×112** 的人脸裁剪图。若使用整张照片，需先用人脸检测（如 YOLOv5-Face）裁出人脸再送入；示例中的 `test_image1`/`test_image2` 建议使用已裁剪好的 112×112 人脸图。
- **两张图**：必须同时提供两张图像才会计算相似度；未指定时从配置文件的 `test_image1`、`test_image2` 读取。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/arcface.py`）常用参数：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | 默认 `examples/arcface/config/arcface.yaml` |
| `--image1` | 第一张人脸图像路径 | 使用 YAML 中 `test_image1` |
| `--image2` | 第二张人脸图像路径 | 使用 YAML 中 `test_image2` |
| `--threshold` | 相似度阈值，高于此值判定为同一人 | `0.6` |
| `--model-path` | 覆盖 YAML 中的 `model_path` | 无 |

**C++ 示例**：

- 第一个参数：配置文件路径，如 `../examples/arcface/config/arcface.yaml`
- 可选：`--image1`、`--image2`、`--threshold <浮点数>`、`--model-path <路径>`

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

**Python：** 在仓库根目录运行：

```bash
python examples/arcface/python/arcface.py --config examples/arcface/config/arcface.yaml
# 指定两张人脸图
python examples/arcface/python/arcface.py --config examples/arcface/config/arcface.yaml --image1 /path/to/face1.png --image2 /path/to/face2.png
# 自定义相似度阈值
python examples/arcface/python/arcface.py --config examples/arcface/config/arcface.yaml --image1 a.png --image2 b.png --threshold 0.5
```

**C++：** 在 `build` 目录运行：

```bash
./examples/arcface ../examples/arcface/config/arcface.yaml
./examples/arcface ../examples/arcface/config/arcface.yaml --image1 /path/to/face1.png --image2 /path/to/face2.png --threshold 0.6
```

## 5. 故障排查

- **模型未找到**：确认已执行 `bash examples/arcface/scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/arcface/arcface_mobilefacenet_cut.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在仓库根目录执行 `bash scripts/download_assets.sh` 可下载资源。
- **Need two face images**：未在配置文件中配置 `test_image1`/`test_image2` 且未传 `--image1`/`--image2`，请至少二选一。
- **输入尺寸不符**：确保输入图像为人脸裁剪且尺寸为 112×112，或由前置人脸检测得到符合该尺寸的裁剪图。
