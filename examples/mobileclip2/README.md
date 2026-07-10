# MobileCLIP2 示例

MobileCLIP2 图文双塔 embedding（256 视觉 + 77 token 文本，768 维）。BPE 表用仓库 `assets/clip/bpe_merges.txt`（与 YOLO-World 共用，不需下载）。

## 1. 模型与权重

- **模型文件**（缓存目录 `~/.cache/models/vision/mobileclip2/`）：
  - 图像：`image_encoder.onnx`
  - 文本：`text_encoder.onnx`
- **下载**：

```bash
bash examples/mobileclip2/scripts/download_models.sh
```

归档路径：`https://archive.spacemit.com/spacemit-ai/model_zoo/vision/mobileclip2/`

**测试图片**：默认 `~/.cache/assets/image/007_dog.jpg`。在根目录执行 `bash scripts/download_assets.sh`。

## 2. 配置文件（config/mobileclip2.yaml）

| 配置项 | 含义 | 默认 |
|--------|------|------|
| `model_path` | 图像编码 ONNX | `~/.cache/models/vision/mobileclip2/image_encoder.onnx` |
| `test_image` | 默认测试图 | `007_dog.jpg` |
| `default_params.text_model_path` | 文本编码 ONNX | `~/.cache/models/vision/mobileclip2/text_encoder.onnx` |
| `default_params.bpe_merges_path` | CLIP BPE 合并表 | `assets/clip/bpe_merges.txt` |
| `image_size` | 视觉输入 | `[256, 256]` |

文本编码走 `EncodeText` / `svc.encode_text()`。输出为 L2 归一化 embedding，比较时用余弦相似度（`EmbeddingSimilarity` / `embedding_similarity`）。

示例默认 prompt（代码内置，非 yaml）：`a photo of a dog/cat/car`。可用 `--text` 覆盖。

## 3. 命令行参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件 | `examples/mobileclip2/config/mobileclip2.yaml` |
| `--image` | 输入图片 | yaml `test_image` |
| `--text` | 逗号分隔 prompt | `a photo of a dog,a photo of a cat,a photo of a car` |
| `--model-path` | 覆盖图像 `model_path` | 无 |

## 4. 运行示例

```bash
# Python
python examples/mobileclip2/python/mobileclip2.py --config examples/mobileclip2/config/mobileclip2.yaml

# C++（build/ 目录）
./examples/mobileclip2 examples/mobileclip2/config/mobileclip2.yaml
./examples/mobileclip2 examples/mobileclip2/config/mobileclip2.yaml --text "a photo of a cat,a photo of a dog"
```

在狗图 `007_dog.jpg` 上，预期 **dog** 得分最高（与参考 demo 一致，数值可能因 resize 插值略有差异）。

## 5. 故障排查

- **模型未找到**：执行 `download_models.sh`，确认 `~/.cache/models/vision/mobileclip2/` 下两个 ONNX 非空。
- **bpe_merges 打不开**：从仓库根目录运行，或 `bpe_merges_path` 用绝对路径。
- **encode_text 报错**：确认已重装带 `encode_text` 绑定的 `spacemit_vision` wheel。
