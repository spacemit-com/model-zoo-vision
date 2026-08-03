# SuperPoint 示例

局部特征提取示例：对单张 BGR8 图像提取关键点和描述子，通过
`VisionService::Infer()` 返回可供 LightGlue 或其他匹配器复用的
`vision::LocalFeatures`。

## 1. 模型与权重

- **模型类型**：局部特征提取（SuperPoint）
- **默认模型文件**：`~/.cache/models/vision/superpoint/superpoint_512x512_top512_batch1.q.onnx`
- **可选模型文件**：`~/.cache/models/vision/superpoint/superpoint_512x512_top512_batch1.onnx`
- **推理后端**：CPU 预处理、SpaceMIT Execution Provider
- **下载**：在 cv 组件根目录执行：

```bash
bash examples/superpoint/scripts/download_models.sh
```

下载脚本会同时获取默认量化模型和可选 FP32 模型。切换模型时修改
`config/superpoint.yaml` 中的 `model_path`。

**数据（测试图片）**：默认测试图为
`~/.cache/assets/image/016_lightglue1.jpg`。若尚未下载资源，请在
cv 组件根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/superpoint.yaml）

本示例使用的配置文件为 `config/superpoint.yaml`。与本模块强相关的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | ONNX 模型路径 | `~/.cache/models/vision/superpoint/superpoint_512x512_top512_batch1.q.onnx` |
| `test_image` | 默认测试图片路径 | `~/.cache/assets/image/016_lightglue1.jpg` |
| `default_params.num_keypoints` | 输出关键点数量 | `512` |
| `default_params.nms_radius` | 关键点非极大值抑制半径 | `4` |
| `default_params.remove_borders` | 忽略输入图边缘的像素宽度 | `4` |
| `default_params.feature_type` | 特征类型标识，用于匹配器契约校验 | `superpoint` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `4` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

输入图会缩放到模型所需的 `512×512` 灰度输入。返回结果中的关键点坐标会映射回
原图像素坐标系，描述子按 `关键点数量 × descriptor_dim` 展平存储；当前模型的
`descriptor_dim` 为 `256`。

## 3. 命令行 / API 参数（与本模块相关）

本示例仅提供 C++ 程序，位置参数如下：

| 参数 | 说明 | 默认 |
|------|------|------|
| 第 1 个参数 | 配置文件路径 | 必填 |
| 第 2 个参数 | 输入图片路径 | 使用 yaml 中的 `test_image` |

程序调用 `VisionService::Infer()` 获取一个 `vision::LocalFeatures`，
随后调用 `VisionService::Draw()` 绘制关键点。输出文件名固定为当前工作目录下的
`superpoint_keypoints.jpg`。

## 4. 运行示例

在 cv 组件根目录完成编译后执行：

```bash
./build/examples/superpoint examples/superpoint/config/superpoint.yaml
./build/examples/superpoint examples/superpoint/config/superpoint.yaml \
  /path/to/image.jpg
```

成功时程序会打印关键点数量和描述子维度，并生成
`superpoint_keypoints.jpg`。

## 5. 故障排查

- **模型未找到**：执行 `bash examples/superpoint/scripts/download_models.sh`，并检查 yaml 中的 `model_path`。
- **测试图未找到**：在 cv 组件根目录执行 `bash scripts/download_assets.sh`。
- **LightGlue 提示特征不兼容**：确保两侧的 `feature_type`、关键点数量和描述子维度与 LightGlue 配置及权重契约一致。
- **模型创建或推理失败**：确认开发板环境可用 `SpaceMITExecutionProvider`，并查看程序输出的 `LastCreateError()` 或 `LastError()`。
