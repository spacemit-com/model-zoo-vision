# MixFormer 示例

单目标跟踪示例：使用首帧目标框初始化 MixFormer-V2，并在后续视频帧中持续跟踪
同一目标，输出带跟踪框的视频。

## 1. 模型与权重

- **模型类型**：有状态单目标跟踪（MixFormer-V2）
- **默认模型文件**：`~/.cache/models/vision/mixformer/mixformer_v2.q.onnx`
- **推理后端**：CPU crop/resize/normalize、SpaceMIT Execution Provider
- **下载**：在 cv 组件根目录执行：

```bash
bash examples/mixformer/scripts/download_models.sh
```

脚本会将模型下载到上述缓存路径，文件名与
`config/mixformer.yaml` 中的 `model_path` 一致。

**数据（测试视频）**：默认测试视频为
`~/.cache/assets/video/004_bag.avi`。若尚未下载资源，请在
cv 组件根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/mixformer.yaml）

本示例使用的配置文件为 `config/mixformer.yaml`。与本模块强相关的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/mixformer/mixformer_v2.q.onnx` |
| `test_video` | 默认测试视频路径 | `~/.cache/assets/video/004_bag.avi` |
| `initial_bbox.x/y/w/h` | 首帧初始框，采用 xywh 像素坐标 | `315 / 140 / 116 / 121` |
| `default_params.update_interval` | 在线模板提交更新的帧间隔 | `200` |
| `default_params.update_threshold` | 候选在线模板的最低跟踪分数 | `0.5` |
| `default_params.max_score_decay` | 历史最佳分数的逐帧衰减系数 | `1.0` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `4` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

示例会把 yaml 中的 `initial_bbox` 从 xywh 转换为
`VisionServiceRequest.initial_bbox` 使用的 xyxy 坐标。固定模板和在线模板输入均为
`112×112`，搜索区域输入为 `224×224`。

## 3. 命令行 / API 参数（与本模块相关）

本示例仅提供 C++ 程序，位置参数如下：

| 参数 | 说明 | 默认 |
|------|------|------|
| 第 1 个参数 | 配置文件路径 | 必填 |
| 第 2 个参数 | 输入视频路径 | 使用 yaml 中的 `test_video` |

首帧请求设置 `has_initial_bbox=true` 并传入初始框；后续帧复用同一个
`VisionService` 实例且只传图像。再次提供初始框会重置跟踪状态。
输出文件名固定为当前工作目录下的 `mixformer_tracking.mp4`。

## 4. 运行示例

在 cv 组件根目录完成编译后执行：

```bash
./build/examples/mixformer examples/mixformer/config/mixformer.yaml
./build/examples/mixformer examples/mixformer/config/mixformer.yaml \
  /path/to/video.mp4
```

自定义视频仍使用 yaml 中的 `initial_bbox`，运行前应按该视频首帧修改初始框。
成功时程序会打印处理帧数，并生成 `mixformer_tracking.mp4`。

## 5. 故障排查

- **模型未找到**：执行 `bash examples/mixformer/scripts/download_models.sh`，并检查 yaml 中的 `model_path`。
- **测试视频未找到**：在 cv 组件根目录执行 `bash scripts/download_assets.sh`。
- **首帧跟踪目标不正确**：`initial_bbox` 使用 xywh 像素坐标，必须位于输入视频首帧范围内。
- **无法生成 MP4**：确认当前 OpenCV 构建支持 `mp4v` 编码，并且当前目录可写。
- **模型创建或推理失败**：确认开发板环境可用 `SpaceMITExecutionProvider`，并查看程序输出的 `LastCreateError()` 或逐帧错误。
