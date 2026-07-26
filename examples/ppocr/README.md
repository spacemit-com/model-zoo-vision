# PP-OCRv6 示例

两阶段 OCR 示例：文本**检测**（DBNet）+ 文本**识别**（CRNN/CTC），对图像输出每行文字的
四边形框 + 识别字符串 + 置信度。

## 1. 模型与权重

- **模型类型**：文本检测 + 识别（PP-OCRv6 tiny / small）
- **下载**（仓库根目录执行）：

```bash
bash examples/ppocr/scripts/download_models.sh
```

会下载到 `~/.cache/models/vision/ppocr/`：

| 文件 | 用途 | 输入 |
|------|------|------|
| `PP-OCRv6_tiny_det_640x640.fp16.onnx` | tiny 检测（默认） | `[1,3,640,640]` |
| `PP-OCRv6_tiny_rec_48x320.dynq.onnx` | tiny 识别（默认） | `[1,3,48,320]` |
| `PP-OCRv6_small_det_640x640.fp16.onnx` | small 检测 | `[1,3,640,640]` |
| `PP-OCRv6_small_rec_48x320.dynq.onnx` | small 识别 | `[1,3,48,320]` |

默认配置 `config/ppocr.yaml` 使用 **tiny**。若改用 small，同步改 `model_path`、`rec_model_path` 和 `dict_path`。

### 字符字典

字典随仓库提供，无需下载：

| 识别模型 | 字典 |
|----------|------|
| tiny rec | `assets/labels/ppocrv6_tiny_dict.txt` |
| small rec | `assets/labels/ppocrv6_small_dict.txt` |

- CTC blank 为索引 0。若字典首行不是 `blank`，加载时会自动在前面插入 `blank`；并按 PaddleOCR `use_space_char` 在末尾追加空格类。
- 字典与 rec 模型类别数不匹配时，创建阶段会直接报错。

**测试图片**：`~/.cache/assets/image/015_ocr.jpg`。若未下载资源，在仓库根目录执行 `bash scripts/download_assets.sh`。

## 2. 配置文件说明（config/ppocr.yaml）

| 配置项 | 含义 | 默认 |
|--------|------|------|
| `model_path` | 检测 ONNX | `~/.cache/models/vision/ppocr/PP-OCRv6_tiny_det_640x640.fp16.onnx` |
| `test_image` | 默认测试图 | `~/.cache/assets/image/015_ocr.jpg` |
| `default_params.rec_model_path` | 识别 ONNX | `~/.cache/models/vision/ppocr/PP-OCRv6_tiny_rec_48x320.dynq.onnx` |
| `default_params.dict_path` | 字符字典（仓库相对路径） | `assets/labels/ppocrv6_tiny_dict.txt` |
| `default_params.det_input_h` / `det_input_w` | 固定检测输入高/宽（>0：等比 resize 后 pad 到 HxW） | `640` / `640` |
| `default_params.det_limit_side_len` | 动态检测长边上限（仅当 `det_input_h/w` 为 0） | `960` |
| `default_params.det_db_thresh` | DB 概率图二值化阈值 | `0.3` |
| `default_params.det_db_box_thresh` | 框平均概率过滤阈值 | `0.6` |
| `default_params.det_db_unclip_ratio` | 文本框膨胀系数 | `1.4` |
| `default_params.rec_img_h` | 识别输入高度 | `48` |
| `default_params.rec_img_w_max` | 识别输入最大宽度（不足则灰底 padding） | `320` |
| `default_params.providers` | 执行提供方 | `SpaceMITExecutionProvider` |

## 3. 命令行参数

**Python**（`python/ppocr.py`）：`--config` / `--image` / `--output` / `--model-path`。  
**C++**（`build` 下）：第一个参数是配置路径，可选 `--image` / `--output` / `--model-path`。

结果字段：`.text`（识别文字）、`.polygon`（四点框，`.x`/`.y`）、`.score`（识别置信度）。

## 4. 运行示例

**Python：**

```bash
cd examples/ppocr/python
python ppocr.py --config ../config/ppocr.yaml --image ~/.cache/assets/image/015_ocr.jpg --output ppocr_result.jpg
```

**C++：** 在 `build` 目录下：

```bash
./examples/ppocr examples/ppocr/config/ppocr.yaml \
  --image ~/.cache/assets/image/015_ocr.jpg \
  --output ppocr_result.jpg
```

切换到 small 时，在 yaml 中改为：

```yaml
model_path: ~/.cache/models/vision/ppocr/PP-OCRv6_small_det_640x640.fp16.onnx
default_params:
  rec_model_path: ~/.cache/models/vision/ppocr/PP-OCRv6_small_rec_48x320.dynq.onnx
  dict_path: assets/labels/ppocrv6_small_dict.txt
```

## 5. 已知限制与故障排查

- **中文叠加渲染**：`cv::putText` 只支持 ASCII，画到图上的中文会显示为 `?`；识别结果本身在打印输出和 `.text` 里是正确的。
- **字典不匹配**：确认 `dict_path` 与当前 rec 模型一致（tiny / small 字典不能混用）。
- **模型未找到**：确认已执行 `bash examples/ppocr/scripts/download_models.sh`，四个 ONNX 都在 `~/.cache/models/vision/ppocr/`。
- **Python 拿不到 `.text`/`.polygon`**：需安装含 Text 字段绑定的 `spacemit_vision` wheel，改动后重新编译安装。
