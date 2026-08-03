# LightGlue 示例

局部特征匹配示例：先使用独立的 SuperPoint `VisionService` 从两张图像提取
`vision::LocalFeatures`，再将两组特征交给 LightGlue
`VisionService::Infer()`，输出 `vision::FeatureMatch` 对应关系。

## 1. 模型与权重

- **模型类型**：局部特征匹配（LightGlue）
- **默认模型文件**：`~/.cache/models/vision/lightglue/lightglue_for_superpoint_512_depth1.fp16.onnx`
- **可选模型文件**：`~/.cache/models/vision/lightglue/lightglue_for_superpoint_512_depth9.fp16.onnx`
- **前端依赖**：默认示例使用 `examples/superpoint/config/superpoint.yaml`
- **推理后端**：CPU 预处理、SpaceMIT Execution Provider
- **下载**：在 cv 组件根目录执行：

```bash
bash examples/superpoint/scripts/download_models.sh
bash examples/lightglue/scripts/download_models.sh
```

LightGlue 本身不负责从图像提取特征，也不固定只能使用 SuperPoint；但当前随附权重
要求输入两组 `feature_type=superpoint`、各 `512` 个关键点、描述子维度为 `256`
的特征。替换其他前端时，必须同时提供与该前端契约匹配的 LightGlue 权重和配置。

**数据（测试图片）**：默认测试图为
`~/.cache/assets/image/016_lightglue1.jpg` 和
`~/.cache/assets/image/017_lightglue2.jpg`。若尚未下载资源，请在
cv 组件根目录执行：

```bash
bash scripts/download_assets.sh
```

## 2. 配置文件说明（config/lightglue.yaml）

本示例使用的配置文件为 `config/lightglue.yaml`。与本模块强相关的字段如下：

| 配置项 | 含义 | 默认或示例 |
|--------|------|------------|
| `model_path` | LightGlue ONNX 模型路径 | `~/.cache/models/vision/lightglue/lightglue_for_superpoint_512_depth1.fp16.onnx` |
| `superpoint_config_path` | 示例使用的特征提取器配置 | `examples/superpoint/config/superpoint.yaml` |
| `test_image1` | 默认查询图片路径 | `~/.cache/assets/image/016_lightglue1.jpg` |
| `test_image2` | 默认训练图片路径 | `~/.cache/assets/image/017_lightglue2.jpg` |
| `default_params.feature_type` | 输入特征类型契约 | `superpoint` |
| `default_params.num_keypoints` | 每张图要求的关键点数量 | `512` |
| `default_params.descriptor_dim` | 每个关键点的描述子维度 | `256` |
| `default_params.filter_threshold` | 匹配分数过滤阈值 | `0.1` |
| `default_params.num_threads` | ONNX Runtime CPU 线程数 | `4` |
| `default_params.providers` | ONNX Runtime 执行提供方 | `SpaceMITExecutionProvider` |

`feature_type`、`num_keypoints` 和 `descriptor_dim` 必须与特征提取器输出及
LightGlue ONNX 输入 shape 同时一致，否则模型会拒绝推理。

## 3. 命令行 / API 参数（与本模块相关）

本示例仅提供 C++ 程序，位置参数如下：

| 参数 | 说明 | 默认 |
|------|------|------|
| 第 1 个参数 | LightGlue 配置文件路径 | 必填 |
| 第 2、3 个参数 | 两张输入图片路径，必须同时提供 | 使用 yaml 中的 `test_image1`、`test_image2` |

示例程序根据 `superpoint_config_path` 创建第二个 `VisionService` 提取两组特征，
再通过 `VisionServiceRequest.local_features0` 和 `local_features1` 调用
LightGlue。输出文件名固定为当前工作目录下的 `lightglue_matches.jpg`。

## 4. 运行示例

在 cv 组件根目录完成编译后执行：

```bash
./build/examples/lightglue examples/lightglue/config/lightglue.yaml
./build/examples/lightglue examples/lightglue/config/lightglue.yaml \
  /path/to/image1.jpg /path/to/image2.jpg
```

成功时程序会打印匹配数量，并生成左右图拼接及绿色匹配连线组成的
`lightglue_matches.jpg`。

## 5. 故障排查

- **模型未找到**：LightGlue 示例同时依赖 SuperPoint 和 LightGlue 权重，请分别执行两个 `download_models.sh`。
- **测试图未找到**：在 cv 组件根目录执行 `bash scripts/download_assets.sh`。
- **特征契约不兼容**：核对两个配置中的 `feature_type`、`num_keypoints`、`descriptor_dim` 以及实际 ONNX 输入 shape。
- **匹配结果为空**：先确认 SuperPoint 正常输出特征，再根据模型需要检查或调整 `filter_threshold`。
- **模型创建或推理失败**：确认开发板环境可用 `SpaceMITExecutionProvider`，并查看程序输出的创建、特征提取或匹配错误。
