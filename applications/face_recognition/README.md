# Face Recognition Application (buffalo_l)

默认流程：Input Image → Face Detection(det_10g) → Face Alignment(5-point) → Embedding Extraction(w600k_r50) → Gender/Age Classification。

仅当 `enable_landmark106: true` 时，才额外执行 106 关键点模型（`2d106det.onnx`）。

## 模型下载

```bash
bash applications/face_recognition/scripts/download_models.sh
```

模型目录：`~/.cache/models/vision/buffalo_l/`（`det_10g.q.onnx`、`det_10g_fixed.q.onnx`、`w600k_r50.q.onnx`、`genderage.q.onnx`、`2d106det.onnx`）。

## 配置

入口配置：`config/face_recognition.yaml`

| 字段 | 说明 |
|------|------|
| `scrfd_model` | 人脸检测（det_10g） |
| `arcface_model` | w600k_r50 embedding |
| `genderage_model` | 性别年龄 |
| `landmark106_model` | 106 点（可选） |
| `enable_landmark106` | 默认 `false` |
| `recognize_threshold` | 识别阈值，默认 `0.3` |
| `face_db_dir` | 人脸库目录，默认 `~/.cache/face_db` |
| `test_image` | 默认测试图，默认 `001_emotion.jpg` |
| `output_image` | 默认输出图名，默认 `output_face_recognition.jpg` |

## 运行

在仓库根目录跑 Python；C++ 在 `build/` 目录。

默认会读 `config/face_recognition.yaml`，**不用手动传 yaml**（和其他 application 一样）。

```bash
# 默认 analyze（用 yaml 里的 test_image）
./applications/example_face_recognition
python applications/face_recognition/python/example_face_recognition.py

# 指定图片
./applications/example_face_recognition --image /path/to/image.jpg

# 注册 / 识别
./applications/example_face_recognition --register alice --image /path/to/face.jpg
./applications/example_face_recognition --recognize --image /path/to/query.jpg

# 摄像头
./applications/example_face_recognition --use-camera --camera-id 1
./applications/example_face_recognition --use-camera --camera-id 1 --recognize
```

CLI：`--config`、`--image`、`--output`、`--use-camera`、`--camera-id`、`--register`、`--recognize`、`--save-image` / `--no-save-image`；C++ 另有 `--camera-width` / `--camera-height` / `--camera-skip`。也支持位置参数 `[config.yaml] [image] [output]`（可选，用来覆盖默认配置）。

默认保存策略：默认 analyze 保存图片；`--register` / `--recognize` / `--use-camera` 默认不保存（可用 `--save-image` 开启）。无显示环境（如纯 SSH）时 OpenCV GUI 可能失败，属运行环境限制。

## Pipeline

```
Input Image
  ↓
Face Detection (det_10g)
  ↓
Face Alignment (5-point)
  ↓
Embedding Extraction (w600k_r50)
  ↓
Gender/Age Classification

(Only when enable_landmark106=true)
  └─ Face crop + affine normalize → 2d106det (106 landmarks)
```

Embedding 存储格式：`uint64 dim` + `float[dim]`（小端），写入 `face_db_dir/<name>.bin`。

## 开发板验收

对照参考 demo：`mimiwang_llama_onnx_0313/buffalo_l/demo/pics/`，比较检测框、embedding 余弦、gender/age。

若 w600k embedding 与参考不一致，可在 `arcface_buffalo.yaml` 取消注释 `norm_std: 128`。
