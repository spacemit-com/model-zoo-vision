# AVTrack 示例

单目标跟踪示例：使用首帧目标框初始化 AVTrack，并在后续视频帧中复用同一个
`VisionService` 实例持续跟踪，输出带跟踪框的视频。

## 1. 模型与权重

- **模型类型**：有状态单目标跟踪（AVTrack）
- **默认模型文件**：`~/.cache/models/vision/avtrack/avtrack_deit_depth4.q.onnx`
- **可选模型文件**：`~/.cache/models/vision/avtrack/avtrack_deit_depth6.q.onnx`
- **推理后端**：CPU 预处理、SpaceMIT Execution Provider
- **下载**：在 cv 组件根目录执行：

```bash
bash examples/avtrack/scripts/download_models.sh
```

下载脚本会同时获取 DeiT depth-4 和 depth-6 量化模型。默认配置使用 depth-4；
切换模型时修改 `config/avtrack.yaml` 中的 `model_path`。

**数据（测试视频）**：默认测试视频为
`~/.cache/assets/video/004_bag.avi`。若尚未下载资源，请在
cv 组件根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/avtrack.yaml）

本示例使用的配置文件为 `config/avtrack.yaml`。与本模块强相关的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/avtrack/avtrack_deit_depth4.q.onnx` |
| `test_video` | 默认测试视频路径 | `~/.cache/assets/video/004_bag.avi` |
| `initial_bbox.x/y/w/h` | 首帧初始框，采用 xywh 像素坐标 | `315 / 140 / 116 / 121` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `4` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

示例会把 yaml 中的 `initial_bbox` 从 xywh 转换为
`VisionServiceRequest.initial_bbox` 使用的 xyxy 坐标。跟踪结果为
`vision::Tracking`，包含原图坐标系下的目标框和跟踪分数。

## 3. 命令行 / API 参数（与本模块相关）

本示例仅提供 C++ 程序，位置参数如下：

| 参数 | 说明 | 默认 |
|------|------|------|
| 第 1 个参数 | 配置文件路径 | 必填 |
| 第 2 个参数 | 输入视频路径 | 使用 yaml 中的 `test_video` |

首帧请求设置 `has_initial_bbox=true` 并传入初始框；后续帧复用同一个
`VisionService` 实例且只传图像。再次提供初始框会重置跟踪状态。
输出文件名固定为当前工作目录下的 `avtrack_tracking.mp4`。

## 4. 运行示例

在 cv 组件根目录完成编译后执行：

```bash
./build/examples/avtrack examples/avtrack/config/avtrack.yaml
./build/examples/avtrack examples/avtrack/config/avtrack.yaml \
  /path/to/video.mp4
```

自定义视频仍使用 yaml 中的 `initial_bbox`，运行前应按该视频首帧修改初始框。
成功时程序会打印处理帧数，并生成 `avtrack_tracking.mp4`。

## 5. 故障排查

- **模型未找到**：执行 `bash examples/avtrack/scripts/download_models.sh`，并检查 yaml 中的 `model_path`。
- **测试视频未找到**：在 cv 组件根目录执行 `bash scripts/download_assets.sh`。
- **首帧跟踪目标不正确**：`initial_bbox` 使用 xywh 像素坐标，必须位于输入视频首帧范围内。
- **无法生成 MP4**：确认当前 OpenCV 构建支持 `mp4v` 编码，并且当前目录可写。
- **模型创建或推理失败**：确认开发板环境可用 `SpaceMITExecutionProvider`，并查看程序输出的 `LastCreateError()` 或逐帧错误。
