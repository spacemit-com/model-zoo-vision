# ArcFace 示例

人脸识别示例：对两张人脸图像提取特征并计算相似度，用于判断是否为同一人。与 YOLOv5-Face（仅检测人脸框）不同，本示例输出相似度分数。

## 1. 模型与权重

- **模型类型**：人脸识别（特征提取 + 相似度）
- **默认模型文件**：`~/.cache/models/vision/arcface/arcface_mobilefacenet_cut.q.onnx`（与 `config/arcface.yaml` 中 `model_path` 一致）
- **下载**：在本示例目录下执行  
  `bash scripts/download_models.sh`  
  会将模型下载到上述缓存路径。

**数据（测试图片）**：默认使用的两张人脸图 `test_image1`、`test_image2` 指向 `~/.cache/assets/image/003_face0.png`、`004_face1.png`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/arcface.yaml）

本示例使用的配置文件为 `config/arcface.yaml`。与**本模块强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/arcface/arcface_mobilefacenet_cut.q.onnx` |
| `image_size` | 输入尺寸 [高, 宽]，须与模型一致 | `[112, 112]` |
| `test_image1` | 示例用第一张人脸图像路径（相对 cv 根目录或绝对） | `~/.cache/assets/image/003_face0.png` |
| `test_image2` | 示例用第二张人脸图像路径 | `~/.cache/assets/image/004_face1.png` |
| `default_params.num_threads` | 推理线程数 | `4` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

说明：通用字段（如 `class`、`default_params` 结构）与其它示例一致，不在此重复。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/arcface.py`）常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | 默认 `examples/arcface/config/arcface.yaml` |
| `--image1` | 第一张人脸图像路径 | 使用 yaml 中 `test_image1` |
| `--image2` | 第二张人脸图像路径 | 使用 yaml 中 `test_image2` |
| `--threshold` | 相似度阈值，高于此值判定为同一人 | `0.6` |
| `--model-path` | 覆盖 yaml 中的 `model_path` | 无 |

**C++ 示例**（需在 `cv/build` 下先编译）：

- 第一个参数：配置文件路径，如 `examples/arcface/config/arcface.yaml`
- 可选：`--image1`、`--image2`、`--threshold <浮点数>`、`--model-path <路径>`

## 4. 特殊配置与注意事项

- **输入要求**：模型输入为 **112×112** 的人脸裁剪图。若使用整张照片，需先用人脸检测（如 YOLOv5-Face）裁出人脸再送入；示例中的 `test_image1`/`test_image2` 建议使用已裁剪好的 112×112 人脸图。
- **两张图**：必须同时提供两张图像才会计算相似度；未指定时从 config 的 `test_image1`、`test_image2` 读取。

## 5. 运行示例

**Python：**

```bash
cd examples/arcface/python
python arcface.py --config ../config/arcface.yaml
# 指定两张人脸图
python arcface.py --config ../config/arcface.yaml --image1 /path/to/face1.png --image2 /path/to/face2.png
# 自定义相似度阈值
python arcface.py --config ../config/arcface.yaml --image1 a.png --image2 b.png --threshold 0.5
```

**C++：** 在 `cv/build` 目录下：

```bash
./examples/arcface examples/arcface/config/arcface.yaml
./examples/arcface examples/arcface/config/arcface.yaml --image1 /path/to/face1.png --image2 /path/to/face2.png --threshold 0.6
```

## 6. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/arcface/arcface_mobilefacenet_cut.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。
- **Need two face images**：未在 config 中配置 `test_image1`/`test_image2` 且未传 `--image1`/`--image2`，请至少二选一。
- **输入尺寸不符**：确保输入图像为人脸裁剪且尺寸为 112×112，或由前置人脸检测得到符合该尺寸的裁剪图。
