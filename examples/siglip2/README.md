# SigLIP2 示例

图文双塔 embedding（224 视觉 + 64 token 文本，768 维）。对输入图与一组文本 prompt 做余弦相似度匹配（zero-shot）。

## 1. 模型与权重

- **模型文件**（与 `config/siglip2.yaml` 一致）：
  - 视觉：`~/.cache/models/vision/siglip2/siglip2_vision_encoder_fp16_proj_dynq.onnx`
  - 文本：`~/.cache/models/vision/siglip2/siglip2_text_encoder_dynq.onnx`
  - 分词器：`~/.cache/models/vision/siglip2/tokenizer.bin`（Gemma BPE 二进制，随模型归档下载）
- **下载**：

```bash
bash examples/siglip2/scripts/download_models.sh
```

若归档暂无 `tokenizer.bin`，可用维护脚本从本机 HuggingFace 缓存导出后上传归档：

```bash
pip install transformers
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/siglip2-base-patch16-224')"
python3 examples/siglip2/scripts/export_tokenizer_bin.py \
  --hf-model google/siglip2-base-patch16-224 \
  --output ~/.cache/models/vision/siglip2/tokenizer.bin
```

**场景标签（可选）**：仓库内 `assets/labels/siglip2_scene_labels.txt`（30 类场景 prompt）。通过 `--labels` 或在 yaml 中设置 `scene_labels_path` 使用。

**测试图片**：默认 `~/.cache/assets/image/007_dog.jpg`。在根目录执行 `bash scripts/download_assets.sh` 下载。

## 2. 配置文件（config/siglip2.yaml）

| 配置项 | 含义 | 默认 |
|--------|------|------|
| `model_path` | 视觉 ONNX | `siglip2_vision_encoder_fp16_proj_dynq.onnx` |
| `test_image` | 默认测试图 | `007_dog.jpg` |
| `scene_labels_path` | 标签文件（可选；未设且无 `--text`/`--labels` 时用内置 3 类） | 无 |
| `default_params.text_model_path` | 文本 ONNX | `siglip2_text_encoder_dynq.onnx` |
| `default_params.tokenizer_path` | tokenizer.bin | `~/.cache/models/vision/siglip2/tokenizer.bin` |
| `image_size` | 视觉输入 | `[224, 224]` |

文本编码走 `EncodeText` / `svc.encode_text()`。

## 3. 命令行参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件 | `examples/siglip2/config/siglip2.yaml` |
| `--image` | 输入图片 | yaml `test_image` |
| `--text` | 逗号分隔 prompt，覆盖其他标签来源 | 无 |
| `--labels` | 指定标签文件路径 | 无（可回退 yaml `scene_labels_path`） |
| `--topk` | 输出前 K 个匹配（C++/Python） | `5` |
| `--model-path` | 覆盖视觉 `model_path` | 无 |

**标签优先级**：`--text` > `--labels` / yaml `scene_labels_path` > 内置 fallback（`a photo of a dog/cat/car`）。

## 4. 运行示例

```bash
# 默认：狗图 + 内置 dog/cat/car
python examples/siglip2/python/siglip2.py --config examples/siglip2/config/siglip2.yaml
./examples/siglip2 examples/siglip2/config/siglip2.yaml

# 30 类场景分类（建议换场景图）
python examples/siglip2/python/siglip2.py --config examples/siglip2/config/siglip2.yaml \
  --labels assets/labels/siglip2_scene_labels.txt --image /path/to/scene.jpg
./examples/siglip2 examples/siglip2/config/siglip2.yaml \
  --labels assets/labels/siglip2_scene_labels.txt --image /path/to/scene.jpg --topk 3

# 自定义 prompt
./examples/siglip2 examples/siglip2/config/siglip2.yaml --text "a photo of a dog,a photo of a cat"
```

## 5. 故障排查

- **模型未找到**：执行 `download_models.sh`。
- **tokenizer.bin 失败**：确认归档已上传；或按上文用 `export_tokenizer_bin.py` 从 HF 缓存生成。
- **场景分类结果不合理**：默认狗图不适合场景标签；请换场景图或用 `--text` 指定类别。
- **推理异常**：若仅 SigLIP2 报错而其他模型正常，可能是推理引擎兼容问题，可反馈引擎侧。
