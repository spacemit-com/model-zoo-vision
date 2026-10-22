# AVTrack 示例

单目标跟踪示例：使用首帧目标框初始化 AVTrack，并在后续视频帧中复用同一个
`VisionService` 实例持续跟踪，输出带跟踪框的视频。

## 1. 模型与权重

- **模型类型**：有状态单目标跟踪（AVTrack）
- **默认模型文件**：`~/.cache/models/vision/avtrack/avtrack_deit_depth4.q.onnx`
- **可选模型文件**：`~/.cache/models/vision/avtrack/avtrack_deit_depth6.q.onnx`
- **推理后端**：CPU 预处理、SpaceMIT Execution Provider
- **下载**：在仓库根目录执行：

```bash
bash examples/avtrack/scripts/download_models.sh
```

下载脚本会同时获取 DeiT depth-4 和 depth-6 量化模型。默认配置使用 depth-4；
切换模型时修改 `config/avtrack.yaml` 中的 `model_path`。

**数据（测试视频）**：默认测试视频为
`~/.cache/assets/video/004_bag.avi`。若尚未下载资源，请在
仓库根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/avtrack.yaml）

本示例使用的配置文件为 `config/avtrack.yaml`。与本示例强相关的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/avtrack/avtrack_deit_depth4.q.onnx` |
| `test_video` | 默认测试视频路径 | `~/.cache/assets/video/004_bag.avi` |
| `initial_bbox.x/y/w/h` | 首帧初始框，采用 xywh 像素坐标 | `315 / 140 / 116 / 121` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `4` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `[SpaceMITExecutionProvider]` |

示例会把 YAML 中的 `initial_bbox` 从 xywh 转换为
`VisionServiceRequest.initial_bbox` 使用的 xyxy 坐标。跟踪结果为
`vision::Tracking`，包含原图坐标系下的目标框和跟踪分数。

## 3. 命令行 / API 参数（与本示例相关）

**Python 示例**（`python/avtrack.py`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `examples/avtrack/config/avtrack.yaml` |
| `--video` | 输入视频路径 | YAML 中的 `test_video` |
| `--bbox X1 Y1 X2 Y2` | 首帧目标框，使用原图像素坐标；注意与 YAML 的 xywh 格式不同 | YAML 中的 `initial_bbox`，自动转换为 xyxy |
| `--output` | 结果视频路径 | `avtrack_tracking.mp4` |
| `--max-frames` | 处理帧数上限，0 表示处理完整视频 | `0` |
| `--model-path` | 覆盖主模型路径 | 无 |

**C++ 示例**：位置参数如下：

| 参数 | 说明 | 默认值 |
|------|------|------|
| 第 1 个参数 | 配置文件路径 | 必填 |
| 第 2 个参数 | 输入视频路径 | 使用 YAML 中的 `test_video` |

首帧请求设置 `has_initial_bbox=true` 并传入初始框；后续帧复用同一个
`VisionService` 实例且只传图像。再次提供初始框会重置跟踪状态。
C++ 输出文件名固定为当前工作目录下的 `avtrack_tracking.mp4`；
Python 可通过 `--output` 指定输出路径。

Python 首帧调用 `service.track(frame, initial_bbox_xyxy)`，后续帧调用
`service.track(frame)`，始终复用同一个服务实例，并通过 `service.draw(frame)` 绘图。

## 4. 运行示例

Python 示例需安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包；
C++ 示例需按仓库 README 编译。

**Python：** 在仓库根目录运行：

```bash
python examples/avtrack/python/avtrack.py --config examples/avtrack/config/avtrack.yaml
python examples/avtrack/python/avtrack.py --config examples/avtrack/config/avtrack.yaml \
  --video /path/to/video.mp4 --bbox 315 140 431 261 --max-frames 100 --output result.mp4
```

**C++：** 在 `build` 目录运行：

```bash
./examples/avtrack ../examples/avtrack/config/avtrack.yaml
./examples/avtrack ../examples/avtrack/config/avtrack.yaml \
  /path/to/video.mp4
```

自定义视频需要与首帧对应的初始框。C++ 示例读取 YAML 的 `initial_bbox`；
Python 也读取该默认值，或通过 `--bbox X1 Y1 X2 Y2` 覆盖。
两个示例均只保存结果视频，不弹出窗口，可在无桌面的 SSH 环境运行。
成功时程序会打印处理帧数，默认生成 `avtrack_tracking.mp4`；
Python 指定 `--output` 时保存到指定路径。

## 5. 故障排查

- **Python 扩展导入失败**：安装与当前平台及 Python 版本匹配的 `spacemit_vision` 包。
- **模型未找到**：执行 `bash examples/avtrack/scripts/download_models.sh`，并检查 YAML 中的 `model_path`。
- **测试视频未找到**：在仓库根目录执行 `bash scripts/download_assets.sh`。
- **首帧跟踪目标不正确**：`initial_bbox` 使用 xywh 像素坐标，必须位于输入视频首帧范围内。
- **无法生成 MP4**：确认当前 OpenCV 构建支持 `mp4v` 编码，并且当前目录可写。
- **模型创建或推理失败**：确认开发板环境可用 `SpaceMITExecutionProvider`，并查看程序输出的 `LastCreateError()` 或逐帧错误。
