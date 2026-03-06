# Emotion 情绪识别示例

对单张人脸图像（或已裁剪的人脸图）进行情绪分类，输出情绪类别与置信度。

## 1. 模型与权重

- **模型类型**：情绪识别（基于 ResNet 等的人脸情绪分类）
- **默认模型文件**：`~/.cache/models/vision/resnet/emotion_resnet50_final.q.onnx`（与 `config/emotion.yaml` 中 `model_path` 一致）
- **下载**：在本示例目录下执行  
  `bash scripts/download_models.sh`  
  会将模型下载到上述缓存路径。

**数据（测试图片）**：默认测试图 `test_image` 指向 `~/.cache/assets/image/003_face0.png`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行示例。

## 2. 配置文件说明（config/emotion.yaml）

本示例使用的配置文件为 `config/emotion.yaml`。与**本模块强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/resnet/emotion_resnet50_final.q.onnx` |
| `test_image` | 默认测试图片路径（建议为人脸图） | `~/.cache/assets/image/003_face0.png` |
| `label_file_path` | 情绪类别标签文件 | `assets/labels/emotion.txt` |
| `image_size` | 输入尺寸 [高, 宽] | `[224, 224]` |
| `default_params.num_threads` | 推理线程数 | `4` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

说明：通用字段（如 `class`、`default_params` 结构）与其它示例一致，不在此重复。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/emotion.py`）常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | 默认 `examples/emotion/config/emotion.yaml` |
| `--image` | 输入图片路径（建议为人脸裁剪图） | 使用 yaml 中 `test_image` |
| `--output` | 输出图片路径 | `result_emotion.jpg` |
| `--model-path` | 覆盖 yaml 中的 `model_path` | 无 |

**C++ 示例**（需在 `cv/build` 下先编译）：

- 第一个参数：配置文件路径，如 `examples/emotion/config/emotion.yaml`
- 可选：`--image`、`--output`、`--model-path`

## 4. 运行示例

**Python：**

```bash
cd examples/emotion/python
python emotion.py --config ../config/emotion.yaml
python emotion.py --config ../config/emotion.yaml --image /path/to/face.png --output result_emotion.jpg
```

**C++：** 在 `cv/build` 目录下：

```bash
./examples/emotion examples/emotion/config/emotion.yaml
./examples/emotion examples/emotion/config/emotion.yaml --image /path/to/face.png --output result_emotion.jpg
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/resnet/emotion_resnet50_final.q.onnx`。
- **测试图片未找到**：默认图片在 `~/.cache/assets/image/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。
- **输入建议**：若使用整张照片，建议先用人脸检测裁出人脸再送入；或使用已裁剪的 224×224 人脸图。
