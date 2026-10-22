# RTMPose 示例

单人 2D 姿态估计示例：输入图像及一个人体框，输出 COCO 顺序的 17 个人体关键点并保存骨架图。
模型不检测人体；多人场景需要先检测人体，再逐人调用。

## 1. 模型与权重

- **模型类型**：自顶向下的人体姿态估计（RTMPose）
- **默认模型文件**：`~/.cache/models/vision/rtmpose/rtmpose_m.fp16.onnx`。
- **可选模型**：`rtmpose_s.fp16.onnx`，可通过 `--model-path` 切换。
- **下载**：在仓库根目录执行：

```bash
bash examples/rtmpose/scripts/download_models.sh
bash scripts/download_assets.sh
```

默认测试图为 `~/.cache/assets/image/013_pose.jpg`。配置中的人体框
`[277, 40, 500, 390]` 对应该图中的滑雪者，只有使用默认测试图且没有传入 `--bbox` 时才生效。

## 2. 配置文件说明（config/rtmpose.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/rtmpose/rtmpose_m.fp16.onnx` |
| `test_image` | 默认测试图片 | `~/.cache/assets/image/013_pose.jpg` |
| `test_bbox` | 示例使用的人体框，顺序为 x1、y1、x2、y2 | `[277, 40, 500, 390]` |
| `default_params.conf_threshold` | 姿态平均分的最低值 | `0.25` |
| `default_params.num_threads` | 推理线程数 | `8` |
| `default_params.providers` | 执行提供方，单元素列表 | `[SpaceMITExecutionProvider]` |

输入尺寸为 192×256（宽×高）。示例将人体框扩张 1.25 倍，再保持宽高比裁出单人图，
推理后将关键点还原到原图坐标。超出图像的部分补黑。

关键点分数是模型原始 SimCC 响应值，可能大于 1；姿态分数是 17 个关键点分数的平均值。
`conf_threshold` 不是限制在 [0, 1] 的概率阈值，应按所用模型及数据选择。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/rtmpose.py`）：

| 参数 | 说明 | 默认值 |
|------|------|------|
| `--config` | 配置文件路径 | `examples/rtmpose/config/rtmpose.yaml` |
| `--image` | 输入图片路径 | YAML 中的 `test_image` |
| `--bbox X1 Y1 X2 Y2` | 原图坐标下的人体框 | 默认图使用 `test_bbox`，自定义图未传时使用整图 |
| `--output` | 输出骨架图路径 | `rtmpose_result.jpg` |
| `--model-path` | 覆盖主模型路径 | 无 |

**C++ 示例**：第一个参数是配置文件路径，其他选项与上表中的
`--image`、`--bbox`、`--output`、`--model-path` 一致。

API 可通过 `VisionService::Infer()` 或 Python `service.infer_image()` 直接提交单人图像，
返回的关键点坐标相对传入图像。若自行裁图或做仿射变换，需要自行把坐标映射回原图。
`test_bbox` 由示例程序读取，不是每次 API 推理自动执行的人体检测或裁剪参数。

## 4. 运行示例

运行前需按仓库 README 编译 C++ 示例。Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。

**Python：** 在仓库根目录运行：

```bash
python examples/rtmpose/python/rtmpose.py --config examples/rtmpose/config/rtmpose.yaml
python examples/rtmpose/python/rtmpose.py --config examples/rtmpose/config/rtmpose.yaml --image /path/to/image.jpg --bbox 50 20 300 450 --output result.jpg
```

**C++：** 在 `build` 目录运行：

```bash
./examples/rtmpose ../examples/rtmpose/config/rtmpose.yaml
./examples/rtmpose ../examples/rtmpose/config/rtmpose.yaml --image /path/to/image.jpg --bbox 50 20 300 450 --output result.jpg
./examples/rtmpose ../examples/rtmpose/config/rtmpose.yaml --model-path ~/.cache/models/vision/rtmpose/rtmpose_s.fp16.onnx
```

示例图只绘制关键点和骨架，终端输出姿态分数和人体框。
更换图片时不会使用默认测试图的人体框；建议为目标人物传入对应 `--bbox`。

## 5. 故障排查

- **模型未找到**：执行模型下载脚本，或用 `--model-path` 指定兼容权重。
- **测试图片未找到**：执行 `bash scripts/download_assets.sh`，或用 `--image` 指定图片。
- **人物关键点异常**：确认输入是单人图片，或用 `--bbox` 指定此人的人体框。
- **多人图仅有一组关键点**：该模型不含人体检测，需要逐人裁剪并分别推理。
- **Python 扩展导入失败**：安装当前平台对应的 `spacemit_vision` 包及其依赖。
