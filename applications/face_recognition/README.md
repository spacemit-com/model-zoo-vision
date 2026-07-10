# Face Recognition Application (buffalo_l)

默认流程：Input Image → Face Detection(det_10g) → Face Alignment(5-point) → Embedding Extraction(w600k_r50) → Gender/Age Classification。

仅当 `enable_landmark106: true` 时，才额外执行 106 关键点模型（`2d106det.onnx`）。

## 模型下载

```bash
bash applications/face_recognition/scripts/download_models.sh
```

模型目录：`~/.cache/models/vision/buffalo_l/`（`det_10g.q.onnx`、`w600k_r50.q.onnx`、`genderage.onnx`、`2d106det.onnx`）。

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

### 默认模式（analyze）

不做人脸注册/识别，只跑流程并默认保存结果图。

```bash
# Python
python applications/face_recognition/python/example_face_recognition.py analyze /path/to/image.jpg
python applications/face_recognition/python/example_face_recognition.py /path/to/image.jpg

# C++（在 build 目录）
./applications/example_face_recognition analyze /path/to/image.jpg
./applications/example_face_recognition /path/to/image.jpg
```

### 注册

```bash
python applications/face_recognition/python/example_face_recognition.py register alice /path/to/face.jpg
./applications/example_face_recognition register alice /path/to/face.jpg
```

### 识别

```bash
python applications/face_recognition/python/example_face_recognition.py recognize /path/to/query.jpg
./applications/example_face_recognition recognize /path/to/query.jpg
```

### 摄像头

```bash
# Python：camera [<camera_index>]，默认不做 embedding
python applications/face_recognition/python/example_face_recognition.py camera 1 --enable-recognition

# C++：支持 camera 子命令，以及与其他 application 对齐的 --use-camera 写法
./applications/example_face_recognition camera 1 --enable-recognition
./applications/example_face_recognition --use-camera --camera-id 1 --camera-width 640 --camera-height 480 --camera-skip 1
```

> 说明：`--use-camera` / `--camera-id` / `--camera-width` / `--camera-height` / `--camera-skip` 仅 C++ 支持；Python 用 `camera [<index>]`。无显示环境（如纯 SSH）时 OpenCV GUI 可能失败，属运行环境限制。

### 常用参数

- `--config <app.yaml>`
- `--output <jpg>`
- `--save-image` / `--no-save-image`
- `--enable-recognition` / `--disable-recognition`
- C++ 额外：`--use-camera` / `--camera-id` / `--camera-width` / `--camera-height` / `--camera-skip`
- `--output` 为相对路径时，保存到当前工作目录（与其他 applications 一致）

默认保存策略：
- `analyze`：默认保存图片
- `register/recognize`：默认不保存（可用 `--save-image` 开启）
- `camera`：实时显示，不保存图片；默认不做 embedding（`--enable-recognition` 后才做）

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
