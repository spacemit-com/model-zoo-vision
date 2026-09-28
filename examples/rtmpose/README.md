# RTMPose 示例

RTMPose 是自顶向下的单人 2D 姿态估计模型。示例接收原图和一个人体框，输出 COCO 顺序的 17 个人体关键点。模型本身不检测人体；多人图像需分别指定每个人的框并逐框推理。

## 1. 模型与测试资源

本示例默认使用 `rtmpose_m.fp16.onnx`，路径为 `~/.cache/models/vision/rtmpose/rtmpose_m.fp16.onnx`。也可通过 `--model-path` 选择 `rtmpose_s.fp16.onnx`；两者具有相同的输入输出契约。

在仓库根目录执行：

```bash
bash examples/rtmpose/scripts/download_models.sh
bash scripts/download_assets.sh
```

前一个脚本只从远端 `vision/rtmpose/` 下载 S/M 两个 FP16 模型，已有缓存则跳过；可通过 `BASE_URL` 覆盖远端地址。

默认测试图片为 `~/.cache/assets/image/013_pose.jpg`。配置中的 `test_bbox: [277, 40, 500, 390]` 仅对应这张图片中的滑雪者，只有使用默认图片且未指定 `--bbox` 时才生效。

## 2. 配置与模型契约

配置文件为 `config/rtmpose.yaml`。主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `model_path` | 默认 ONNX 文件；可被 `--model-path` 覆盖 |
| `test_image`、`test_bbox` | 默认演示图片和单人框 |
| `default_params.conf_threshold` | 17 个关键点原始 SimCC 分数平均值的最低值；不是概率阈值 |
| `default_params.num_threads` | CPU 线程数；默认为 8 |
| `default_params.providers` | ONNX Runtime provider；默认 SpaceMIT EP |

ONNX 输入为 RGB、NCHW、FP32 `[1, 3, 256, 192]`，按 ImageNet 均值和标准差归一化。输出 `simcc_x` 为 `[1, 17, 384]`、`simcc_y` 为 `[1, 17, 512]`。RTMPose 实现可接收任意尺寸的单人 BGR 图：内部将整张输入图保持宽高比仿射到 192×256（不额外扩框，空白区域补黑），解码后把关键点反算回输入图坐标；192×256 输入不再重复仿射。

C++/Python 示例按指定人体框扩张 1.25 倍，保持宽高比后直接从原图做一次仿射采样；只有超出原图的区域才补黑。示例传入的图已经是 192×256，所以模型内部走恒等路径，随后示例再将关键点还原到原图坐标并绘图。

示例结果图只绘制关键点和骨架，不绘制人体框或分数标签；终端仍输出姿态分数和人体框坐标。

关键点分数取 X/Y 两路 SimCC 峰值的较小值，不裁剪到 `[0, 1]`；它是模型原始响应值，可能大于 1。`Pose.score` 为 17 个关键点分数的平均值。

RTMPose 模型不包含人体检测。未指定人体框时，示例把整张图当成一个人处理；直接传多人图只能得到一个姿态，且可能不准确。`VisionService::Infer` / Python `infer_image` 可直接传单人图或人体裁剪图，返回坐标相对传入图；如果传入的是整张多人图，模型不会自动逐人检测和裁剪。

直接推理返回的 `Pose.bbox` 是整张输入图；如果像示例一样先从原图仿射裁出一个人，再要在原图上绘制，则需将它替换为原图人体框，并把关键点逆变换回原图。

## 3. 运行

在 `build/` 目录中运行 C++ 示例：

```bash
./examples/rtmpose ../examples/rtmpose/config/rtmpose.yaml
./examples/rtmpose ../examples/rtmpose/config/rtmpose.yaml --image /path/to/image.jpg --bbox 50 20 300 450 --output result.jpg
./examples/rtmpose ../examples/rtmpose/config/rtmpose.yaml --model-path ../rtmpose_s.fp16.onnx
```

使用已安装的 `spacemit_vision` Python 包运行。Python 示例需要 NumPy、OpenCV 和 PyYAML，这些依赖由该包的安装配置声明：

```bash
python examples/rtmpose/python/rtmpose.py
python examples/rtmpose/python/rtmpose.py --image /path/to/image.jpg --bbox 50 20 300 450 --output result.jpg
```

运行前需构建 C++ 示例或安装当前源码构建的 Python 包。`--image` 指向自定义图片时不会使用配置中的默认 `test_bbox`；没有 `--bbox` 时，整张图片被视为一个人的人体框。

## 4. 排查

- 模型找不到：先运行模型脚本，或用 `--model-path` 指向已有 ONNX 文件。
- 测试图找不到：运行 `scripts/download_assets.sh`，或用 `--image` 指定图片。
- 人物关键点位置异常：确认输入是单人图片，或通过 `--bbox` 指定此人的框。
