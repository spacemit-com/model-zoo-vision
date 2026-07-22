# AdaFace 示例

高精度人脸 embedding 示例（IR-101，512 维）。对两张人脸图提取特征并计算余弦相似度，用于判断是否为同一人。与 ArcFace（MobileFaceNet）相比，AdaFace 模型更大、精度更高，阈值需单独标定。

## 1. 模型与权重

- **模型类型**：人脸 embedding（112×112 center-crop，512 维）
- **默认模型**：`~/.cache/models/vision/adaface/adaface_ir101_webface12m_merged.dynq.onnx`
- **下载**：

```bash
bash examples/adaface/scripts/download_models.sh
```

**测试图片**：默认 `test_image1` / `test_image2` 指向 `~/.cache/assets/image/003_face0.png`、`004_face1.png`。若尚未下载，在**组件根目录**执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件（config/adaface.yaml）

| 配置项 | 含义 | 默认 |
|--------|------|------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/adaface/adaface_ir101_webface12m_merged.dynq.onnx` |
| `image_size` | 输入尺寸 [高, 宽] | `[112, 112]` |
| `test_image1` / `test_image2` | 示例人脸图路径 | `003_face0.png` / `004_face1.png` |
| `default_params.num_threads` | 推理线程数 | `4` |
| `default_params.providers` | 执行提供方 | `SpaceMITExecutionProvider` |

## 3. 命令行参数

**Python**（`python/adaface.py`）：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件 | `examples/adaface/config/adaface.yaml` |
| `--image1` / `--image2` | 两张人脸图 | yaml 中 `test_image1` / `test_image2` |
| `--threshold` | 相似度阈值 | `0.35`（AdaFace 模型，非 ArcFace 的 0.6） |
| `--model-path` | 覆盖 `model_path` | 无 |

**C++**（在 `build/` 目录编译后）：

```bash
./examples/adaface examples/adaface/config/adaface.yaml
./examples/adaface examples/adaface/config/adaface.yaml --image1 a.png --image2 b.png --threshold 0.35
```

## 4. 注意事项

- 模型输入为 **112×112** 人脸区域。示例图已是裁剪好的人脸；整张照片需先用 YOLOv5-Face 等检测再裁剪。
- 预处理为 **center-crop**（与 ArcFace 对齐方式可能不同），对比分数时请使用同一 pipeline。
- 默认阈值 `0.35` 为示例参考值，生产环境请按业务数据标定。

## 5. 运行示例

```bash
# Python
python examples/adaface/python/adaface.py --config examples/adaface/config/adaface.yaml

# C++（build/ 目录）
./examples/adaface examples/adaface/config/adaface.yaml
```

## 6. 故障排查

- **模型未找到**：执行 `bash examples/adaface/scripts/download_models.sh`。
- **测试图片未找到**：在根目录执行 `bash scripts/download_assets.sh`。
- **Need two face images**：配置或命令行需同时提供两张图。
