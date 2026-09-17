# SCRFD 示例

人脸检测示例：输出人脸框、置信度和五个人脸关键点，不包含人脸识别、对齐或属性分析。

## 1. 模型与权重

- **模型类型**：人脸检测（SCRFD，buffalo_l det_10g）
- **默认模型文件**：`~/.cache/models/vision/buffalo_l/det_10g_fixed.q.onnx`
- **下载**：在本示例目录执行 `bash scripts/download_models.sh`。

本示例与 `applications/face_recognition` 共用 `buffalo_l` 权重缓存，
下载脚本只获取检测模型，不下载识别、年龄性别等模型。

默认测试图为 `~/.cache/assets/image/006_test.jpg`。若尚未下载资源，
在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

### 模型输入输出说明

复用现有 `ScrfdDetector`：将 BGR 图片转为 RGB，等比缩放并放在输入画布
左上角，空白区域补零，按 `(pixel - 127.5) / 128` 归一化为 NCHW。
输入尺寸从模型读取；默认权重为 640×640。

当前实现按三个尺度读取分数、框偏移和五点偏移，共9个输出。
`--model-path` 仅适用于兼容此输出契约的权重，不保证支持任意 SCRFD 导出。
解码和 NMS 后，框与关键点坐标映射回原图。

公共 API 使用 `vision::Pose` 表示每张人脸的框及五点，推理意图为
`kEstimatePose`，并非人体姿态估计。Python 的 `infer_image()` 返回同样的信息。
绘图显示人脸框、分数和五点，不连接人体骨架；不需要 COCO 标签文件。

## 2. 配置文件说明（config/scrfd.yaml）

| 配置项 | 含义 | 默认 |
|--------|------|------|
| `model_path` | 检测模型路径 | `~/.cache/models/vision/buffalo_l/det_10g_fixed.q.onnx` |
| `test_image` | 默认测试图片 | `~/.cache/assets/image/006_test.jpg` |
| `default_params.conf_threshold` | 人脸置信度阈值 | `0.5` |
| `default_params.nms_threshold` | NMS IoU 阈值 | `0.4` |
| `default_params.num_threads` | 推理线程数 | `8` |
| `default_params.providers` | 执行提供方 | `SpaceMITExecutionProvider` |

CPU 对照可将 provider 改为 `CPUExecutionProvider`。
示例使用 BGR 图片和 CPU 预处理，不提供摄像头、MPP 或 OpenCL 参数。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/scrfd.py`）：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | YAML 配置路径 | 示例内的 `config/scrfd.yaml` |
| `--model-path` | 覆盖模型路径 | 无 |
| `--image` | 输入图片 | YAML 中的 `test_image` |
| `--output` | 输出图片 | `scrfd_result.jpg` |

**C++ 示例**：第一个参数为 YAML 配置路径，可选 `--model-path`、`--image`、
`--output`、`--help`。显式传入的相对路径按当前工作目录解析。

两个示例都通过公共服务执行推理和绘图，并打印人脸数量、置信度和框坐标。
未检出人脸时数量为0，仍保存原图，不视为运行失败。

## 4. 运行示例

**Python：** 先按仓库 README 安装当前构建的 `spacemit_vision` 包及 OpenCV 等依赖。
从仓库根目录进入示例目录：

```bash
cd examples/scrfd/python
python3 scrfd.py --config ../config/scrfd.yaml
python3 scrfd.py --config ../config/scrfd.yaml \
  --image /path/to/face.jpg --output result.jpg
```

**C++：** 按仓库 README 构建并启用 examples 后，在仓库的 `build/` 目录执行：

```bash
./examples/scrfd ../examples/scrfd/config/scrfd.yaml
./examples/scrfd ../examples/scrfd/config/scrfd.yaml \
  --image /path/to/face.jpg --output result.jpg
```

只传配置即可运行默认图片，结果保存到当前工作目录。

## 5. 故障排查

- **模型未找到**：执行本示例的 `scripts/download_models.sh`；缓存目录是 `buffalo_l`，不是 `scrfd`。
- **图片未找到**：执行仓库根目录的 `scripts/download_assets.sh`，或指定 `--image`。
- **没有检出人脸**：检查图片内容、人脸大小和遮挡情况，可在 YAML 中适当调低置信度阈值。
- **输出契约错误**：确认使用默认的 `det_10g_fixed.q.onnx`，而非其他模型或不同输出排列的导出。
- **Python 导入或运行库错误**：确认安装的原生扩展与当前 `libvision.so` 匹配。
