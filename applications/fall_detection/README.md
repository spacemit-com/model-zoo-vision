# 跌倒检测应用

对视频/摄像头进行人体姿态估计 + STGCN 动作识别：提取人体关键点序列，由 STGCN 判断是否发生“跌倒”（Fall Down），输出检测结果与可视化。

## 1. 模型与权重

- **模型类型**：姿态估计（YOLOv8-Pose）+ 动作识别（STGCN/TSSTG，30 帧骨架序列，含 Fall Down 等 7 类）
- **默认模型文件**：
  - 姿态模型：`~/.cache/models/vision/yolov8_pose/yolov8n-pose.q.onnx`（`pose_model_path`）
  - STGCN 模型：`~/.cache/models/vision/stgcn/stgcn.fp32.onnx`（`stgcn_model_path`，由 `config/stgcn_action.yaml` 引用）
- **下载**：在本应用目录下执行 `bash scripts/download_models.sh` 会将姿态模型与 STGCN 模型下载到缓存路径。

**数据（测试视频）**：默认测试视频 `test_video` 指向 `~/.cache/assets/video/002_fall.mp4`。若尚未下载资源，请在 **cv 组件根目录** 执行：

```bash
bash scripts/download_assets.sh
```

脚本会将 `image/`、`video/` 等资源下载到 `~/.cache/assets/`，之后即可直接运行应用。

## 2. 配置文件说明

本应用涉及两个配置：**应用配置** `config/fall_detection.yaml` 与 **STGCN 模型配置** `config/stgcn_action.yaml`（由 model_factory 按 `stgcn_model_name` 加载）。

**config/fall_detection.yaml** 与**本应用强相关**的字段：`pose_config_path`、`pose_model_path`、`test_video`、`label_file_path`、`kp_threshold`、`stgcn_model_name`、`stgcn_model_path`、`stgcn_wait_frames`、`stgcn_smooth_window`。**config/stgcn_action.yaml**：`model_path`、`class`、`default_params` 等，供 create_model 创建 StgcnActionRecognizer。

## 3. 命令行 / API 参数（与本应用相关）

**Python**（`python/example_fall_detection.py`）常用参数：`--config`、`--video`、`--use-camera`、`--camera-id`、`--conf-threshold`、`--iou-threshold`、`--num-threads`、`--kp-threshold`、`--pose-model`、`--stgcn-model`、`--stgcn-wait-frames`、`--smooth-window`。

**C++**（需在 `cv/build` 下先编译）：第一个参数为应用配置路径，可选 `--config`、`--video`、`--use-camera`、`--camera-id`、`--kp-threshold`、`--pose-model`、`--stgcn-model`。

## 4. 运行示例

**Python：** 建议在 **cv 组件根目录** 运行：

```bash
python applications/fall_detection/python/example_fall_detection.py
python applications/fall_detection/python/example_fall_detection.py --config applications/fall_detection/config/fall_detection.yaml --video /path/to/video.mp4
python applications/fall_detection/python/example_fall_detection.py --config applications/fall_detection/config/fall_detection.yaml --use-camera
```

**C++：** 在 `cv/build` 目录下：

```bash
./applications/example_fall_detection applications/fall_detection/config/fall_detection.yaml
./applications/example_fall_detection applications/fall_detection/config/fall_detection.yaml --video /path/to/video.mp4
./applications/example_fall_detection applications/fall_detection/config/fall_detection.yaml --use-camera
```

## 5. 故障排查

- **模型未找到**：确认已执行 `scripts/download_models.sh`（在 `applications/fall_detection/` 下），且 `pose_model_path`、`stgcn_model_path` 指向的路径存在。
- **测试视频未找到**：默认视频在 `~/.cache/assets/video/` 下。在 cv 根目录执行 `bash scripts/download_assets.sh` 可下载资源。
- **STGCN 输入**：STGCN 需要 30 帧、13 关键点（COCO 子集）的骨架序列；应用内部会按 `stgcn_wait_frames` 积累帧后推理。
