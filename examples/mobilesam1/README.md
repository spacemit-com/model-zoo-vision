# MobileSAM1 示例

提示分割示例：根据图像中的点或框提示分割目标，输出掩膜叠加图和绿色提示框。

## 1. 模型与权重

- **模型类型**：提示分割（MobileSAM Tiny，encoder + decoder）
- **默认 encoder**：`~/.cache/models/vision/mobilesam1/mobilesam_encoder_tiny_sim.fp16.onnx`
- **默认 decoder**：`~/.cache/models/vision/mobilesam1/mobilesam_decoder_sim.dynq.onnx`
- **下载**：在本示例目录下执行 `bash scripts/download_models.sh`，从服务器的 `vision/mobilesam1/` 下载到上述缓存目录。

**数据（测试图片）**：默认 `test_image` 指向 `~/.cache/assets/image/008_picture.jpg`。
若尚未下载资源，请在仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

### 模型格式说明

当前支持输入为 float32 `[1,3,448,448]` 的 encoder 和配套 decoder。
图像等比缩放后在右下补黑色，再按 RGB 均值和标准差归一化。
Decoder 根据提示返回4组候选掩膜，选择预测 IoU 最高的一组，裁除补边并还原
原图尺寸，按 logit > 0 生成二值掩膜。当前接口不支持历史掩膜反馈。

## 2. 配置文件说明（config/mobilesam1.yaml）

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | Encoder 路径 | `~/.cache/models/vision/mobilesam1/mobilesam_encoder_tiny_sim.fp16.onnx` |
| `default_params.decoder_model_path` | Decoder 路径 | `~/.cache/models/vision/mobilesam1/mobilesam_decoder_sim.dynq.onnx` |
| `test_image` | 默认测试图片 | `~/.cache/assets/image/008_picture.jpg` |
| `default_params.num_threads` | 每个 Session 的线程数 | `4` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

CPU 对照可将 provider 改为 `CPUExecutionProvider`。模型使用两个 Session，
默认各4线程；在8个 A100 的板上，不要同时给两个 EP Session 各分配8线程。

## 3. 命令行 / API 参数（与本模块相关）

**Python 示例**（`python/mobilesam1.py`）：

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 配置文件路径 | 必填 |
| `--image` | 输入图片路径 | 使用 YAML 中的 `test_image` |
| `--output` | 输出图片路径 | `mobilesam1_result.jpg` |
| `--box` | 原图像素坐标下的 `x1 y1 x2 y2` | `190 70 460 280` |

**C++ 示例**：使用位置参数 `config [image [output [x1 y1 x2 y2]]]`。
只传配置即可运行；图片、输出和提示框默认值与 Python 一致。更换图片时，
应同时传入对应目标的提示框。

API 支持点和框：C++ 请求填写 `point_coords`、`point_labels`，Python 调用
`infer_image_points(image, points, labels)`。标签为0背景点、1前景点、
2框左上角、3框右下角、-1填充点；仅输入点时自动追加填充点。
输出复用 `Segmentation`，其中 `score` 是预测 IoU，`label=0` 不代表语义类别。

## 4. 运行示例

**Python：** 需安装包含 MobileSAM 接口的 `spacemit_vision` 包及 OpenCV、NumPy。
从仓库根目录进入示例目录：

```bash
cd examples/mobilesam1/python
python3 mobilesam1.py --config ../config/mobilesam1.yaml
python3 mobilesam1.py --config ../config/mobilesam1.yaml \
  --image /path/to/image.jpg --box 190 70 460 280 --output result.jpg
```

**C++：** 按仓库 README 构建并启用 examples 后，在仓库的 `build/` 目录下：

```bash
./examples/mobilesam1 ../examples/mobilesam1/config/mobilesam1.yaml
./examples/mobilesam1 ../examples/mobilesam1/config/mobilesam1.yaml \
  /path/to/image.jpg result.jpg 190 70 460 280
```

输出保存到当前工作目录。两个示例均沿用原独立 C++ 示例的 BGR
`(30,144,144)`、50%透明度叠加掩膜，并绘制绿色提示框，只保存一张结果图。
绿色框是输入提示，不是模型检测出的框。

## 5. 故障排查

- **模型未找到**：在本示例目录执行 `bash scripts/download_models.sh`，确认 encoder 和 decoder 路径均正确。
- **测试图片未找到**：在仓库根目录执行 `bash scripts/download_assets.sh`，确认 `008_picture.jpg` 已下载。
- **旧库没有模型或点提示接口**：重新构建并安装 Python 包；C++ 调用程序也需要随更新后的公共请求结构重新编译。
- **提示参数错误**：点和标签数量需一致，框用标签2、3依次表示左上角和右下角，并确保 `x1 < x2`、`y1 < y2`。
- **结果与预期目标不符**：检查提示框是否对应当前图片中的目标。默认框仅适用于样例图片。
