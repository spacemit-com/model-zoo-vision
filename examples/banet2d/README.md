# BANet2D 示例

双目深度示例：输入一对同尺寸 BGR8 左右图，通过
`VisionService::Infer()` 计算视差，并输出原始左图坐标系下的彩色视差图。

## 1. 模型与权重

- **模型类型**：双目视差估计（BANet2D）
- **默认模型文件**：`~/.cache/models/vision/banet2d/banet_2d_small_512x640_resize_mode_nearest_slice_sim.q.onnx`
- **推理后端**：CPU 预处理、SpaceMIT Execution Provider
- **下载**：在 cv 组件根目录执行：

```bash
bash examples/banet2d/scripts/download_models.sh
```

脚本会将模型下载到上述缓存路径，文件名与
`config/banet2d.yaml` 中的 `model_path` 一致。

**数据（双目测试图）**：默认左右图分别为
`~/.cache/assets/image/018_banet2d_left.png` 和
`~/.cache/assets/image/019_banet2d_right.png`。若尚未下载资源，请在
cv 组件根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/banet2d.yaml）

本示例使用的配置文件为 `config/banet2d.yaml`。与本模块强相关的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/banet2d/banet_2d_small_512x640_resize_mode_nearest_slice_sim.q.onnx` |
| `test_image1` | 默认左图路径 | `~/.cache/assets/image/018_banet2d_left.png` |
| `test_image2` | 默认右图路径 | `~/.cache/assets/image/019_banet2d_right.png` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `4` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

左右图必须具有相同的宽高。模型内部会保持宽高比缩放并填充到模型输入尺寸，
最终返回与原始左图同尺寸的 `vision::Disparity`，其中视差图为
`CV_32FC1`，数值单位为像素。

## 3. 命令行 / API 参数（与本模块相关）

本示例仅提供 C++ 程序，位置参数如下：

| 参数 | 说明 | 默认 |
|------|------|------|
| 第 1 个参数 | 配置文件路径 | 必填 |
| 第 2、3 个参数 | 左图和右图路径，必须同时提供 | 使用 yaml 中的 `test_image1`、`test_image2` |

程序通过 `VisionServiceRequest.image` 和 `image2` 提交双目图像，
`VisionService::Infer()` 返回一个 `vision::Disparity`。输出文件名固定为
当前工作目录下的 `banet2d_disparity.png`。

## 4. 运行示例

在 cv 组件根目录完成编译后执行：

```bash
./build/examples/banet2d examples/banet2d/config/banet2d.yaml
./build/examples/banet2d examples/banet2d/config/banet2d.yaml \
  /path/to/left.png /path/to/right.png
```

成功时程序会打印视差图尺寸，并生成伪彩色视差图
`banet2d_disparity.png`。

## 5. 故障排查

- **模型未找到**：执行 `bash examples/banet2d/scripts/download_models.sh`，并检查 yaml 中的 `model_path`。
- **测试图未找到**：在 cv 组件根目录执行 `bash scripts/download_assets.sh`。
- **左右图尺寸不一致**：输入图必须具有完全相同的宽高，否则推理会明确失败。
- **模型创建或推理失败**：确认开发板环境可用 `SpaceMITExecutionProvider`，并查看程序输出的 `LastCreateError()` 或 `LastError()`。
