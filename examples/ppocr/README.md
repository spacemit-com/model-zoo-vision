# PP-OCRv5 示例

两阶段 OCR 示例：文本**检测**（DBNet）+ 文本**识别**（CRNN/CTC），对图像输出每行文字的
四边形框 + 识别字符串 + 置信度。

## 1. 模型与权重

- **模型类型**：文本检测 + 识别（PP-OCRv5 mobile）
- **模型文件**（与 `config/ppocr.yaml` 一致）：
  - 检测：`~/.cache/models/vision/ppocr/PP-OCRv5_mobile_det.onnx`（输入 `[N,3,H,W]` 动态，输出概率图）
  - 识别：`~/.cache/models/vision/ppocr/PP-OCRv5_mobile_rec.onnx`（输入 `[N,3,48,W]`，输出 `[N,T,18385]` CTC）
- **下载**：在本示例目录下执行 `bash scripts/download_models.sh`
  （从 `archive.spacemit.com` 下载 det/rec 两个 ONNX 到上述缓存路径）。

### 字符字典

识别模型输出 18385 个类别索引，靠 `assets/labels/ppocr_keys.txt` 把索引映射成文字
（PP-OCRv5 mobile rec 字典，18384 行，随仓库提供）。

- 约定：字典**第 0 行就是 `blank`**（CTC blank，索引 0）。代码逐行读入、**不额外插 blank**
  （否则整体错位、识别出乱码）；空行代表空格；按 PaddleOCR `use_space_char` 末尾再补一个空格类。
- 若文件缺失或为空，创建模型时会**明确报错**（不静默）。

## 2. 配置文件说明（config/ppocr.yaml）

| 配置项 | 含义 | 默认 |
|--------|------|------|
| `model_path` | 检测 ONNX | `~/.cache/models/vision/ppocr/PP-OCRv5_mobile_det.onnx` |
| `test_image` | 默认测试图 | `~/.cache/assets/image/015_ocr.jpg` |
| `default_params.rec_model_path` | 识别 ONNX | `~/.cache/models/vision/ppocr/PP-OCRv5_mobile_rec.onnx` |
| `default_params.dict_path` | 字符字典（仓库相对路径） | `assets/labels/ppocr_keys.txt` |
| `default_params.det_limit_side_len` | 检测输入长边上限（缩放到 32 倍数） | `960` |
| `default_params.det_db_thresh` | DB 概率图二值化阈值 | `0.3` |
| `default_params.det_db_box_thresh` | 框平均概率过滤阈值 | `0.6` |
| `default_params.det_db_unclip_ratio` | 文本框膨胀系数 | `2.0` |
| `default_params.rec_img_h` | 识别输入高度 | `48` |
| `default_params.rec_img_w_max` | 识别输入最大宽度（不足则灰底 padding） | `320` |
| `default_params.providers` | 执行提供方 | `CPUExecutionProvider` |

## 3. 命令行参数

**Python 示例**（`python/ppocr.py`）：`--config` / `--image` / `--output` / `--model-path`。
**C++ 示例**（`cv/build` 下编译）：第一个参数是配置路径，可选 `--image` / `--output` / `--model-path`。

结果对象字段：`.text`（识别文字）、`.polygon`（四点框，`.x`/`.y`）、`.score`（识别置信度）。

## 4. 运行示例

**Python：**

```bash
cd examples/ppocr/python
python ppocr.py --config ../config/ppocr.yaml --image /path/to/text.jpg --output ppocr_result.jpg
```

**C++：** 在 `cv/build` 目录下：

```bash
./examples/ppocr examples/ppocr/config/ppocr.yaml --image /path/to/text.jpg --output ppocr_result.jpg
```

## 5. 已知限制与故障排查

- **中文叠加渲染**：`cv::putText` 只支持 ASCII，画到图上的中文会显示为 `?`；但识别文字本身
  正确（在打印输出和结果 `.text` 里）。需要图上正确显示中文可后续接 FreeType 渲染。
- **字典缺失/不匹配**：创建即报错。确认 `assets/labels/ppocr_keys.txt` 存在、且与 PP-OCRv5
  rec 的类别数匹配（否则识别出乱码）。
- **检测/识别精度**：本实现按 PaddleOCR 标准算法写（DB 二值化 + unclip、CTC 贪心解码），
  建议在开发板上用真实文本图与 PaddleOCR 官方结果对照 det 框与 rec 文字。
- **模型未找到**：确认 `scripts/download_models.sh` 已执行，det/rec 两个 ONNX 都在
  `~/.cache/models/vision/ppocr/` 下。
- **Python 拿不到 `.text`/`.polygon`**：pybind 扩展需含本次新增的 Text 字段绑定，改动后要重编
  安装 wheel。
