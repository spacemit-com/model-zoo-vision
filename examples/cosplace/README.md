# CosPlace 示例

视觉地点识别（Visual Place Recognition）示例：从图像中提取全局描述符，用于地点检索和定位任务。

## 1. 模型与权重

- **模型类型**：地点识别（Place Recognition / Global Descriptor Extraction）
- **默认模型文件**：`~/.cache/models/vision/cosplace/cosplace_resnet50_512.q.onnx`（与 `config/cosplace.yaml` 中 `model_path` 一致）
- **下载**：在本示例目录下执行  
  `bash scripts/download_models.sh`  
  会将模型下载到上述缓存路径。

**数据（测试图片）**：CosPlace 适用于室外场景地点识别。示例使用巴塞罗那街景图像集：
- **测试数据**：`~/.cache/assets/image/barcelona/`（包含 2 张查询图像和 5 张数据库图像）
- **下载**：在本示例目录下执行  
  `bash scripts/download_assets.sh`  
  会将测试数据下载到上述缓存路径

## 2. 配置文件说明（config/cosplace.yaml）

本示例使用的配置文件为 `config/cosplace.yaml`。与**本模块强相关**的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/cosplace/cosplace_resnet50_512.q.onnx` |
| `test_query` | 默认查询图像路径 | `~/.cache/assets/image/barcelona/query/query_0.jpg` |
| `test_database` | 默认数据库目录 | `~/.cache/assets/image/barcelona/database` |
| `top_k` | 返回前 K 个最相似的地点 | `5` |
| `image_size` | 输入尺寸 [高, 宽]，仅作说明；实际输入尺寸由 ONNX 模型推导 | `[512, 512]` |
| `default_params.num_threads` | 推理线程数 | `8` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

说明：通用字段（如 `class`、`default_params` 结构）与其它示例一致，不在此重复。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/cosplace.py`）常用参数：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | 必需 |
| `--query` | 查询图像路径（覆盖配置文件中的 `test_query`） | 从配置读取 |
| `--database` | 数据库图像目录（覆盖配置文件中的 `test_database`） | 从配置读取 |
| `--top-k` | 返回前 K 个最相似的地点（覆盖配置文件中的 `top_k`） | 从配置读取，默认 `5` |
| `--model-path` | 覆盖 yaml 中的 `model_path` | 从配置读取 |

**C++ 示例**（需在 `vision/build` 下先编译）：

- 第一个参数：配置文件路径，如 `../examples/cosplace/config/cosplace.yaml`
- 可选参数（覆盖配置文件）：`--query <path>`、`--database <dir>`、`--top-k <n>`、`--model-path <path>`

## 4. 特殊配置与注意事项

- **输入要求**：模型输入为 **512×512** 的 RGB 图像，使用 ImageNet 归一化。
- **检索模式**：模型输出 L2 归一化的全局描述符（512 维），使用 L2 距离进行检索。
- **数据库**：需要提前准备参考地点图像数据库，模型会在其中检索与查询图像最相似的地点。

## 5. 运行示例

**Python：**

```bash
cd examples/cosplace/python
# 使用配置文件中的默认参数
python cosplace.py --config ../config/cosplace.yaml

# 覆盖配置文件参数
python cosplace.py --config ../config/cosplace.yaml \
    --query /path/to/query.jpg \
    --database /path/to/database_images \
    --top-k 3
```

**C++：** 在 `vision/build` 目录下：

```bash
# 使用配置文件中的默认参数
./examples/cosplace ../examples/cosplace/config/cosplace.yaml

# 覆盖配置文件参数
./examples/cosplace ../examples/cosplace/config/cosplace.yaml \
    --query /path/to/query.jpg \
    --database /path/to/database_images \
    --top-k 3
```

## 6. 输出说明

示例程序会：
1. 提取查询图像的全局描述符
2. 遍历数据库中所有图像并提取描述符
3. 计算查询与数据库中每张图像的 L2 距离
4. 返回距离最小的前 K 个匹配结果

输出格式：
```
Top-5 Place Matches (by L2 distance)
========================================
#1 Distance: 0.3214 - /path/to/db/image1.jpg
#2 Distance: 0.4567 - /path/to/db/image2.jpg
...
```

## 7. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`，且 `model_path` 指向 `~/.cache/models/vision/cosplace/cosplace_resnet50_512.q.onnx`。
- **测试数据未找到**：确认已执行 `scripts/download_assets.sh`，或在配置文件中设置正确的 `test_query` 和 `test_database`。
- **Query/Database 未指定**：必须在配置文件中设置 `test_query` 和 `test_database`，或通过命令行参数 `--query` 和 `--database` 指定。
- **数据库目录为空**：确保数据库目录包含图像文件（.jpg、.jpeg、.png）。
- **内存不足**：大规模数据库会占用较多内存，可分批处理或减少数据库大小。
