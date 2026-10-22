# MobileCLIP2 示例

MobileCLIP2 图文双塔 embedding（256 视觉 + 77 token 文本，768 维）。BPE 表用仓库 `assets/clip/bpe_merges.txt`（与 YOLO-World 共用，不需下载）。

## 1. 模型与权重

- **模型文件**（缓存目录 `~/.cache/models/vision/mobileclip2/s3/`，与 `config/mobileclip2.yaml` 一致）：
  - 图像：`image_encoder.fp16.onnx`
  - 文本：`text_encoder.onnx`
- **下载**：在仓库根目录执行：

```bash
bash examples/mobileclip2/scripts/download_models.sh
```

归档路径：`https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobileclip2_s3/`

**测试图片**：默认 `~/.cache/assets/image/007_dog.jpg`。在仓库根目录执行 `bash scripts/download_assets.sh`。

## 2. 配置文件说明（config/mobileclip2.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------|
| `model_path` | 图像编码 ONNX | `~/.cache/models/vision/mobileclip2/s3/image_encoder.fp16.onnx` |
| `test_image` | 默认测试图 | `~/.cache/assets/image/007_dog.jpg` |
| `default_params.text_model_path` | 文本编码 ONNX | `~/.cache/models/vision/mobileclip2/s3/text_encoder.onnx` |
| `default_params.bpe_merges_path` | CLIP BPE 合并表 | `assets/clip/bpe_merges.txt` |
| ONNX 输入尺寸 | 由模型自动读取，不能通过 YAML 改写 | 见模型文件 |

文本编码走 `EncodeText` / `svc.encode_text()`。输出为 L2 归一化 embedding，比较时用余弦相似度（`EmbeddingSimilarity` / `embedding_similarity`）。

示例默认 prompt（代码内置，非 YAML）：`a photo of a dog/cat/car`。可用 `--text` 覆盖。

## 3. 命令行 / API 参数（与本示例相关）

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` |配置文件| `examples/mobileclip2/config/mobileclip2.yaml` |
| `--image` | 输入图片 | YAML `test_image` |
| `--text` | 逗号分隔 prompt | `a photo of a dog,a photo of a cat,a photo of a car` |
| `--model-path` | 覆盖图像 `model_path` | 无 |

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

**Python：** 在仓库根目录运行：

```bash
python examples/mobileclip2/python/mobileclip2.py --config examples/mobileclip2/config/mobileclip2.yaml
python examples/mobileclip2/python/mobileclip2.py --config examples/mobileclip2/config/mobileclip2.yaml --text "a photo of a cat,a photo of a dog"
```

**C++：** 在 `build` 目录运行：

```bash
./examples/mobileclip2 ../examples/mobileclip2/config/mobileclip2.yaml
./examples/mobileclip2 ../examples/mobileclip2/config/mobileclip2.yaml --text "a photo of a cat,a photo of a dog"
```

终端显示输入图片与各条文本的相似度；更换图片时，应使用适合该图片内容的文本候选。

## 5. 故障排查

- **模型未找到**：执行 `examples/mobileclip2/scripts/download_models.sh`，确认 `~/.cache/models/vision/mobileclip2/s3/` 下两个 ONNX 非空。
- **bpe_merges 打不开**：从仓库根目录运行，或 `bpe_merges_path` 用绝对路径。
- **encode_text 报错**：确认已重装带 `encode_text` 绑定的 `spacemit_vision` wheel。
