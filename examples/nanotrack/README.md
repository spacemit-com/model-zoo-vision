# NanoTrack 示例

单目标跟踪示例：首帧使用目标框初始化 NanoTrack，模板特征在后续视频帧中复用，
最终输出带跟踪框的视频。

## 1. 模型与权重

- **模型类型**：有状态单目标跟踪（NanoTrack）
- **模板 backbone**：`~/.cache/models/vision/nanotrack/nanotrack_backbone1.onnx`
- **搜索 backbone**：`~/.cache/models/vision/nanotrack/nanotrack_backbone2.q.onnx`
- **匹配 head**：`~/.cache/models/vision/nanotrack/nanotrack_head.q.onnx`
- **推理后端**：模板 backbone 使用 CPU；搜索 backbone 和 head 使用 SpaceMIT Execution Provider
- **下载**：在 cv 组件根目录执行：

```bash
bash examples/nanotrack/scripts/download_models.sh
```

三个模型共同组成 NanoTrack，运行时缺一不可。下载脚本会将它们放入上述缓存目录，
文件名与 `config/nanotrack.yaml` 完全一致。

**数据（测试视频）**：默认复用 AVTrack 和 MixFormer 的测试视频
`~/.cache/assets/video/004_bag.avi`。若尚未下载资源，请在 cv 组件根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/nanotrack.yaml）

本示例使用的配置文件为 `config/nanotrack.yaml`。与本模块强相关的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | 搜索 backbone 模型路径 | `nanotrack_backbone2.q.onnx` |
| `default_params.template_model_path` | 模板 backbone 模型路径 | `nanotrack_backbone1.onnx` |
| `default_params.head_model_path` | 匹配 head 模型路径 | `nanotrack_head.q.onnx` |
| `test_video` | 默认测试视频路径 | `~/.cache/assets/video/004_bag.avi` |
| `initial_bbox.x/y/w/h` | 首帧初始框，采用 xywh 像素坐标 | `315 / 140 / 116 / 121` |
| `default_params.context_amount` | 模板和搜索区域的上下文比例 | `0.5` |
| `default_params.penalty_k` | 尺寸与宽高比变化惩罚系数 | `0.148` |
| `default_params.window_influence` | Hanning window 对候选分数的影响 | `0.462` |
| `default_params.learning_rate` | 目标框尺寸平滑系数 | `0.390` |
| `default_params.num_threads` | 搜索 backbone 的 CPU 线程数 | `4` |
| `default_params.template_num_threads` | 模板 backbone 的 CPU 线程数 | `4` |
| `default_params.head_num_threads` | 匹配 head 的 CPU 线程数 | `1` |
| `default_params.providers` | 搜索 backbone 和 head 的执行提供方 | `SpaceMITExecutionProvider` |

模板输入为 `[1,3,127,127]`，搜索输入为 `[1,3,255,255]`；head 接收两路特征并
输出 `[1,2,16,16]` 分类 logits 和 `[1,4,16,16]` 边框回归。

## 3. 命令行 / API 参数（与本模块相关）

本示例仅提供 C++ 程序，位置参数如下：

| 参数 | 说明 | 默认 |
|------|------|------|
| 第 1 个参数 | 配置文件路径 | 必填 |
| 第 2 个参数 | 输入视频路径 | 使用 yaml 中的 `test_video` |

首帧请求设置 `has_initial_bbox=true` 并传入 xyxy 初始框；后续帧复用同一个
`VisionService` 实例且只传图像。再次提供初始框会重置模板和跟踪状态。
输出文件名固定为当前工作目录下的 `nanotrack_tracking.mp4`。

## 4. 运行示例

在 cv 组件根目录完成编译后执行：

```bash
./build/examples/nanotrack examples/nanotrack/config/nanotrack.yaml
./build/examples/nanotrack examples/nanotrack/config/nanotrack.yaml \
  /path/to/video.mp4
```

自定义视频仍使用 yaml 中的 `initial_bbox`，运行前应按该视频首帧修改初始框。
成功时程序会打印处理帧数，并生成 `nanotrack_tracking.mp4`。

## 5. 故障排查

- **模型未找到**：执行 `bash examples/nanotrack/scripts/download_models.sh`，并核对三个模型路径。
- **测试视频未找到**：在 cv 组件根目录执行 `bash scripts/download_assets.sh`。
- **首帧跟踪目标不正确**：`initial_bbox` 使用 xywh 像素坐标，必须位于输入视频首帧范围内。
- **无法生成 MP4**：确认当前 OpenCV 构建支持 `mp4v` 编码，并且当前目录可写。
- **模型 shape 不匹配**：确认三个模型来自同一 NanoTrack 工件组，没有混用其他版本。
- **模型创建或推理失败**：确认模板模型可用 CPU 运行，且开发板环境可用 `SpaceMITExecutionProvider`。
