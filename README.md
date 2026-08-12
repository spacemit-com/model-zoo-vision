# Vision 组件

## 1. 项目简介

本组件为 SpacemiT **计算机视觉模型部署库**，基于 ONNX Runtime，在 C++ 与 Python 下提供统一的推理与封装接口，便于集成到机器人、嵌入式与服务器等场景。功能特性如下：

| 类别     | 支持                                                                 |
| -------- | -------------------------------------------------------------------- |
| 模型类型 | 检测、分类、分割、跟踪、人脸、姿态等，一套接口接入                   |
| 后端     | ONNX Runtime（含 SpaceMIT 等定制运行时）                             |
| 接口     | C++（`vision_service.h` 类接口，`cv::Mat` 入参）、Python（`spacemit_vision` wheel：`VisionServiceNative`） |
| 可扩展   | 抽象基类与公共工具（预处理、NMS、绘图等）便于接入新模型             |

## 2. 验证模型

按以下顺序完成依赖安装、模型准备与示例运行。

### 2.1. 安装依赖

- **编译环境**：CMake 3.10+，C++17；Python 3.12+（可选）。
- **C++ 依赖**：OpenCV 4（与本工程 `CMakeLists.txt` 中 `find_package` 一致，含 core、imgproc、imgcodecs、highgui、dnn）、`/opt/eigen-spacemit`、`/opt/openblas-spacemit`、spacemit-ort、yaml-cpp。路径可通过 `OpenCV_DIR`、`SPACEMIT_DIR`、`EIGEN_SPACEMIT_DIR`、`OPENBLAS_DIR` 等在 CMake 中设置。
- **Python 依赖**（跑示例 / 使用 wheel）：NumPy、OpenCV；打 wheel 时需 `pybind11`、`build`（见下方命令）。

```bash
# 系统依赖示例（Linux）
sudo apt-get update
sudo apt-get install -y python3-spacemit-ort opencv-spacemit spacemit-onnxruntime libyaml-cpp-dev \
    eigen-spacemit libeigen3-dev openblas-spacemit
```

安装后 Eigen / OpenBLAS 默认位于 `/opt/eigen-spacemit`、`/opt/openblas-spacemit`（与 `CMakeLists.txt` 一致）。


**Python 安装（推荐先用于跑通示例）：**

Python 接口由 `spacemit_vision` wheel 提供（C++ `VisionService` 绑定）。
`BUILD_PYTHON_BINDINGS` / `BUILD_PYTHON_WHEEL` 默认均为 ON，一条构建命令即可编译并打出 wheel
（需在目标解释器上安装 `pybind11`，如开发板 python3.14）：

```bash
# 1) 一步到位：编译 C++ 核心 + Python 扩展 + 打包 wheel
python3 -m pip install -U pybind11 build setuptools wheel
cmake -S . -B build && cmake --build build -j        # whl 产出在 src/python/dist/

# 2) 安装并自检
python3 -m pip install --force-reinstall src/python/dist/*.whl
python3 -c "from spacemit_vision import VisionServiceNative; print('ok')"
```

如需仅编扩展不打包：`-DBUILD_PYTHON_WHEEL=OFF`；纯 C++：`-DBUILD_PYTHON_BINDINGS=OFF`。

### 2.2. 下载模型

模型统一放在 **`~/.cache/models/vision/<type>/`**（如 `~/.cache/models/vision/yolov8/`）。若运行示例时提示「Model file not found」，请在对应示例目录下执行下载脚本：

```bash
# 下载 examples/yolov8 所需要的模型
bash examples/yolov8/scripts/download_models.sh
```

也可以一键下载所有示例/应用所需模型（在 vision 组件根目录执行）：

```bash
bash scripts/download_all_models.sh
```

该脚本会遍历常见 examples/applications 下的 `download_models.sh`；若某模型（例如 **STGCN**）未包含在脚本列表中，请到对应 `examples/<name>/scripts/` 单独执行下载脚本。

全部下载完成后，目录大致如下（示意，实际模型更多）：

```bash
~/.cache/models/vision/
├── arcface/
│   └── arcface_mobilefacenet_cut.q.onnx
├── adaface/
│   └── adaface_ir101_webface12m_merged.dynq.onnx
├── yolov8/
│   ├── yolov8n_no_dfl.q.onnx
│   ├── yolov8s_no_dfl.q.onnx
│   └── ...
├── yolov11/
│   ├── yolo11n.q.onnx
│   └── ...
├── buffalo_l/
│   ├── det_10g_fixed.q.onnx
│   ├── w600k_r50.q.onnx
│   └── ...
└── ...
```

### 2.3. 下载资源（图片、视频）

示例与应用的默认测试图片、视频统一从 **`~/.cache/assets/`** 读取。首次使用可执行：

```bash
bash scripts/download_assets.sh
```

脚本会从 `https://archive.spacemit.com/spacemit-ai/model_zoo/assets/` 下载 `image`、`video` 等目录到 `~/.cache/assets`（与服务器目录名一致）。配置中默认路径为 `~/.cache/assets/image/`、`~/.cache/assets/video/`。

```bash
bianbu@k3:~/.cache/assets# tree
.
├── image
│   ├── 001_emotion.jpg
│   ├── 002_fire.jpg
│   ├── 003_face0.png
│   ├── 004_face1.png
│   ├── 005_kitten.jpg
│   ├── 006_test.jpg
│   ├── 007_dog.jpg
│   ├── 008_picture.jpg
│   ├── 009_test_unet.jpg
│   ├── 012_gesture.jpg
│   └── 013_pose.jpg
└── video
    ├── 001_crowd.mp4
    ├── 002_fall.mp4
    └── 003_palace.mp4
```

各示例的配置文件与默认路径见各模型子目录 README（如 `examples/yolov8/README.md`、`examples/arcface/README.md`）。

### 2.4. 测试

本节提供示例程序的编译与运行方式，便于开发者快速验证效果。使用前需先按下列两种方式之一完成编译，再运行对应示例。

- **在 SDK 中验证**（2.4.1）：在已拉取的 SpacemiT Robot SDK 工程内用 `mm` 编译，产物部署到 `output/staging`，适合整机集成或与其他模块联调。
- **独立构建下验证**（2.4.2）：在 Vision 组件目录下用 CMake 本地编译，不依赖完整 SDK，适合快速体验或在不使用 repo 的环境下使用。

#### 2.4.1. 在 SDK 中验证

**编译**：环境准备、源码拉取与 Model Zoo 在 SDK 中的编译与集成说明见 SpacemiT 社区文档 [SpacemiT Model Zoo](https://www.spacemit.com/community/document/info?lang=zh&nodepath=software/SDK/bianbu/ai/model-zoo.md)（使用 repo 时需先完成 `repo init`、`repo sync` 等）。

```bash
source build/envsetup.sh
cd components/model_zoo/vision
mm
```

构建产物会安装到 `output/staging`。

**运行**：运行前在 SDK 根目录执行 `source build/envsetup.sh`，使 PATH 与库路径指向 `output/staging`，然后可执行：

**Python 示例（以 YOLOv8 为例）：**

> 运行 Python 示例前需先安装 `spacemit_vision` wheel（见 [2.1](#21-安装依赖)）；
> 示例通过 `from spacemit_vision import VisionServiceNative` 调用 C++ 推理接口。

```bash
cd components/model_zoo/vision/examples/yolov8/python
python yolov8.py --config ../config/yolov8.yaml
python yolov8.py --config ../config/yolov8.yaml --image /path/to/image.jpg --conf-threshold 0.3
```

**C++ 示例：**

```bash
yolov8 config/yolov8.yaml
yolov8 config/yolov8.yaml --image /path/to/image.jpg
```

#### 2.4.2. 独立构建下验证

在 Vision 组件目录下完成编译后，运行下列示例。

**Python 示例（以 YOLOv8 为例）：**

> 同样需先安装 `spacemit_vision` wheel（见 [2.1](#21-安装依赖)）。

```bash
cd examples/yolov8/python
python yolov8.py --config ../config/yolov8.yaml
python yolov8.py --config ../config/yolov8.yaml --image /path/to/image.jpg --conf-threshold 0.3
```

**C++ 示例：**

```bash
# 在组件根目录下进行编译
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 运行示例
./examples/yolov8 examples/yolov8/config/yolov8.yaml
./examples/yolov8 examples/yolov8/config/yolov8.yaml --image /path/to/image.jpg
```

更多模型（ByteTrack、OC-SORT、ArcFace、AdaFace、SigLIP2、MobileCLIP2 等）的用法与参数见**各示例子目录 README**。

## 3. 应用开发

本章说明如何在自有工程中**集成 CV 并调用 API**。环境与依赖见 [2.1](#21-安装依赖)，模型准备见 [2.2](#22-下载模型)，编译与运行示例见 [2.4](#24-测试)。

### 3.1. 构建与集成产物

无论通过 [2.4.1](#241-在-sdk-中验证)（SDK）或 [2.4.2](#242-独立构建下验证)（独立构建）哪种方式编译，完成后**应用开发所需**的头文件与库如下。集成时只需**包含头文件并链接对应库**。

**构建产物说明：**

| 产物 | 路径 / 说明 |
|------|-------------|
| **头文件（集成必选）** | `include/vision_service.h`（仅此 1 个公开头文件）。C++ 类接口，推理与绘制均使用 `cv::Mat`，集成时需链接 OpenCV；业务工程 `#include "vision_service.h"` 并链接 `vision` 及依赖库。 |
| **动态库** | 目标名固定为 `vision`；Linux 常见产物为 `build/libvision.so`。 |

**集成到自有工程时需链接的库（明确清单）**  
`vision` 为动态库，最终可执行文件链接时必须同时链接以下依赖：

| 依赖库 | 说明 |
|--------|------|
| OpenCV | 与本工程一致：`opencv_core`、`opencv_imgproc`、`opencv_imgcodecs`、`opencv_highgui`、`opencv_dnn`（CMake 中 `find_package(OpenCV ... COMPONENTS core imgproc imgcodecs highgui dnn)` 与 `${OpenCV_LIBS}`） |
| ONNX Runtime | `onnxruntime`（Linux 常见文件名：`libonnxruntime.so`） |
| SpaceMIT EP | `spacemit_ep`（Linux 常见文件名：`libspacemit_ep.so`；使用 SpaceMITExecutionProvider 时必需） |
| yaml-cpp | `yaml-cpp`（Linux 常见文件名：`libyaml-cpp.so`） |

**推荐 CMake 链接写法（与本工程一致）：**
```cmake
target_link_libraries(your_app PRIVATE
  vision
  ${OpenCV_LIBS}
  onnxruntime
  spacemit_ep
  yaml-cpp
)
```

**安装布局**（执行 `cmake --install .` 后）：`vision` 安装到 `lib/`，仅 `vision_service.h` 安装到 `include/`；示例可执行文件安装到 `bin/`。  
注意：`onnxruntime` / `spacemit_ep` / `yaml-cpp` / OpenCV 共享库属于外部依赖，不由本项目安装，需要由系统或上层工程提供并保证运行时可搜索到。

**可选 CMake 变量：**

| 变量 | 说明 |
|------|------|
| `OpenCV_DIR` / `OpenCV_INSTALL_DIR` | OpenCV 的 CMake 配置目录或安装根目录 |
| `SPACEMIT_DIR` | SpaceMIT 运行时/头文件根目录 |
| `BUILD_EXAMPLES` | 是否构建示例（默认 ON） |
| `BUILD_TESTS` | 是否构建测试与 `vision_infer_benchmark`（默认 ON） |
| `BUILD_PYTHON_BINDINGS` | 是否编译 Python 扩展 `_vision_service_cpp`（默认 ON；无 pybind11 时跳过） |
| `BUILD_PYTHON_WHEEL` | 默认构建时是否打包 `spacemit_vision` wheel（默认 ON；依赖 `BUILD_PYTHON_BINDINGS` 与 `python -m build`） |

### 3.2. API 使用与 CMake 集成

- **C++**：`#include "vision_service.h"`，包含路径指向本组件 `include/`（或安装后的 `include`）；链接 `vision` 及上表所列依赖库。接口为类 `VisionService`，创建与推理、绘制均使用 `cv::Mat`，无需 raw buffer 转换。
- **Python**：安装 `spacemit_vision` wheel 后，`from spacemit_vision import VisionServiceNative`，用 `VisionServiceNative.create(config_yaml)` 创建服务，再调 `infer_image` / `infer_embedding` / `encode_text` / `infer_sequence` / `draw`；参数见各示例子目录 README。图文双塔模型（SigLIP2、MobileCLIP2）的文本侧用 `encode_text`。

**C++ 调用示例（图像推理 + 绘制）：**

```cpp
#include "vision_service.h"
#include <opencv2/opencv.hpp>

auto service = VisionService::Create(config_path, model_path_override, false);
if (!service) { /* 使用 VisionService::LastCreateError() */ return; }

cv::Mat image = cv::imread(image_path);
VisionServiceResponse response;
if (service->Infer(image, &response) != VISION_SERVICE_OK) { /* 使用 service->LastError() */ return; }

// 遍历结果（variant，按任务分型）
for (const auto& r : response.results) {
    const vision::BoundingBox box = vision::get_bbox(r);
    int label = vision::get_label(r);
    float score = vision::get_score(r);
    // 需要强类型时：if (auto* d = std::get_if<vision::Detection>(&r)) { ... }
    (void)box; (void)label; (void)score;
}

cv::Mat vis;
if (!response.results.empty())
    service->Draw(image, response, &vis);  // 显式传入 response
else
    vis = image;
// 使用 vis 显示或保存
```

**Python 调用示例（图像推理 + 绘制）：**

```python
import cv2
from spacemit_vision import VisionServiceNative, VisionServiceStatus

svc = VisionServiceNative.create("examples/yolov8/config/yolov8.yaml")
img = cv2.imread("test.jpg")

status, results = svc.infer_image(img)            # conf/iou 可选，<=0 用 yaml 默认
if status != VisionServiceStatus.OK:
    raise RuntimeError(svc.last_error())

for r in results:                                  # 按任务分型读取字段
    print(r.label, r.score, (r.x1, r.y1, r.x2, r.y2))
    # 姿态: r.keypoints(.x/.y/.visibility)；分割: r.mask；分类: r.class_scores

if svc.supports_draw():
    st, vis = svc.draw(img)                        # 复用最近一次推理结果
    if st == VisionServiceStatus.OK:
        cv2.imwrite("result.jpg", vis)
```

**图文 embedding（SigLIP2 / MobileCLIP2）示例：**

```python
svc = VisionServiceNative.create("examples/siglip2/config/siglip2.yaml")
img = cv2.imread("~/.cache/assets/image/007_dog.jpg")
st, image_emb = svc.infer_embedding(img)
st, text_emb = svc.encode_text("a photo of a dog")
score = VisionServiceNative.embedding_similarity(image_emb, text_emb)
```

**CMake 集成示例**：将本组件作为子目录添加后，对目标执行 `target_link_libraries(your_app PRIVATE vision ${OpenCV_LIBS} onnxruntime spacemit_ep yaml-cpp)`，并 `target_include_directories(your_app PRIVATE path/to/vision/include)`。

## 4. 常见问题

1. **Python 导入报错 `No module named 'spacemit_vision'`**  
   请按 [2.1](#21-安装依赖) 编译并安装 wheel：`cmake --build build -j` 后 `pip install src/python/dist/*.whl`。

2. **模型文件未找到**  
   运行 `examples/<model>/scripts/download_models.sh` 将模型下载到 `~/.cache/models/vision/<type>/`，详见 2.2。

3. **依赖版本**  
   Python 以 `src/python/setup.py` 中 `install_requires` 为准；C++ 以 CMake 要求及 CI 环境为准；大版本升级若有 ABI/API 变更会在版本与发布中说明。

## 5. 版本与发布

版本以本目录 `CMakeLists.txt` / `src/python/setup.py` 为准。

| 版本   | 说明 |
| ------ | ---- |
| 0.1.0  | 计算机视觉模型部署库，支持 YOLOv8、ByteTrack、OC-SORT、ArcFace、AdaFace、SigLIP2、MobileCLIP2、ResNet 等；C++ / Python 统一接口，ONNX Runtime 后端。 |

重要变更与兼容性说明将随版本更新在本文档或仓库 Release 中记录。

## 6. 贡献方式

欢迎提交 Issue 与 Pull Request。开发前请确认编码风格与现有 C++/Python 风格一致，并通过相关测试。若仓库中存在 **`CONTRIBUTORS.md`**，贡献者与维护者名单以该文件为准。

## 7. License

本组件源码文件头声明为 Apache-2.0，最终以本目录 **`LICENSE`** 文件为准。

## 8. 附录：模型性能

**包含前后处理：**

> 说明：以下数据基于 K1/K3 平台实测（推理引擎 2.0.6），为阶段性信息，持续优化中，请以最新文档为准。

K1:

|  模型大类   |       具体模型        |   输入大小    | 数据类型 | 帧率(4核) |
| :---------: | :-------------------: | :-----------: | :------: | :-------: |
|   resnet    |       resnet50        | [1,3,224,224] |   int8   |   23.7    |
|  mobilenet  |     mobilenet_v1      | [1,3,224,224] |   int8   |   58.2    |
|             |     mobilenet_v2      | [1,3,224,224] |   int8   |   48.5    |
|   arcface   | arcface_mobilefacenet | [1,3,112,112] |   int8   |   110.8   |
| yolov5-face |     yolov5n-face      | [1,3,640,640] |   int8   |    6.8    |
|   yolov8    |        yolov8n        | [1,3,640,640] |   int8   |   11.0    |
|             |        yolov8s        | [1,3,640,640] |   int8   |    5.9    |
|             |        yolov8m        | [1,3,640,640] |   int8   |    3.1    |
| yolov8-pose |     yolov8n-pose      | [1,3,640,640] |   int8   |   11.0    |
|             |     yolov8s-pose      | [1,3,640,640] |   int8   |    5.7    |
|             |     yolov8m-pose      | [1,3,640,640] |   int8   |    3.1    |
| yolov8-seg  |      yolov8n-seg      | [1,3,640,640] |   int8   |    5.0    |
|             |      yolov8s-seg      | [1,3,640,640] |   int8   |    3.1    |
|             |      yolov8m-seg      | [1,3,640,640] |   int8   |    2.0    |
|   yolo11    |        yolo11n        | [1,3,640,640] |   int8   |   11.2    |
|             |        yolo11s        | [1,3,640,640] |   int8   |    5.9    |
|             |        yolo11m        | [1,3,640,640] |   int8   |    2.3    |
|   yolo26    |        yolo26n        | [1,3,640,640] |   int8   |   18.1    |

K3:

|       模型大类        |         具体模型         |    输入大小     |  数据类型  |  帧率(4核)  |  帧率(8核)  |
| :-------------------: | :----------------------: | :-------------: | :--------: | :---------: | :---------: |
|        resnet         |         resnet50         |  [1,3,224,224]  |    int8    |    106.5    |    146.2    |
|       mobilenet       |       mobilenet_v1       |  [1,3,224,224]  |    int8    |    192.8    |    278.7    |
|                       |       mobilenet_v2       |  [1,3,224,224]  |    int8    |    150.0    |    222.7    |
|     efficientnet      |     efficientnet_b0      |  [1,3,224,224]  |    int8    |    59.8     |    74.0     |
|                       |    efficientnet_v2_s     |  [1,3,224,224]  |    int8    |    43.5     |    59.9     |
|          vit          |         vit_b_16         |  [1,3,224,224]  |    int8    |    26.9     |    39.4     |
|        arcface        |  arcface_mobilefacenet   |  [1,3,112,112]  |    int8    |    266.3    |    393.2    |
|        adaface        |      adaface_ir101       |  [1,3,112,112]  |    int8    |    24.1     |    30.1     |
|        emotion        |     emotion_resnet50     |  [1,3,224,224]  |    int8    |    132.9    |    179.3    |
|      yolov5-face      |       yolov5n-face       |  [1,3,640,640]  |    int8    |    29.5     |    37.6     |
|        yolov5         |         yolov5n          |  [1,3,640,640]  |    int8    |    32.5     |    38.5     |
|                       |         yolov5s          |  [1,3,640,640]  |    int8    |    24.8     |    30.9     |
|                       |      yolov5_gesture      |  [1,3,640,640]  |    int8    |    36.2     |    50.6     |
|        yolov8         |         yolov8n          |  [1,3,640,640]  |    int8    |    56.4     |    76.7     |
|                       |         yolov8s          |  [1,3,640,640]  |    int8    |    34.3     |    48.5     |
|                       |         yolov8m          |  [1,3,640,640]  |    int8    |    17.8     |    27.7     |
|      yolov8-pose      |       yolov8n-pose       |  [1,3,640,640]  |    int8    |    50.2     |    67.0     |
|                       |       yolov8s-pose       |  [1,3,640,640]  |    int8    |    31.4     |    43.8     |
|                       |       yolov8m-pose       |  [1,3,640,640]  |    int8    |    18.0     |    26.0     |
|      yolov8-seg       |       yolov8n-seg        |  [1,3,640,640]  |    int8    |    28.4     |    34.6     |
|                       |       yolov8s-seg        |  [1,3,640,640]  |    int8    |    17.1     |    20.6     |
|                       |       yolov8m-seg        |  [1,3,640,640]  |    int8    |    10.2     |    12.7     |
|        yolo11         |         yolo11n          |  [1,3,640,640]  |    int8    |    41.8     |    57.8     |
|                       |         yolo11s          |  [1,3,640,640]  |    int8    |    28.0     |    39.7     |
|                       |         yolo11m          |  [1,3,640,640]  |    int8    |    14.0     |    21.0     |
|        yolo12         |         yolo12n          |  [1,3,640,640]  |    int8    |    16.8     |    20.6     |
|        yolo26         |         yolo26n          |  [1,3,640,640]  |    int8    |    41.2     |    55.3     |
|       bytetrack       |      yolov8n(19目标)       |  [1,3,640,640]  |    int8    |    40.2     |    51.0     |
|                       |      yolov8s(22目标)       |  [1,3,640,640]  |    int8    |    27.3     |    35.9     |
|        ocsort         |      yolov8n(19目标)       |  [1,3,640,640]  |    int8    |    24.5     |    28.6     |
|                       |      yolov8s(22目标)       |  [1,3,640,640]  |    int8    |    18.9     |    22.6     |


**复现方法**：

参照 [2.4.2](#242-独立构建下验证) 完成 C++ 构建（需开启 `BUILD_TESTS`，默认 ON）。`vision_infer_benchmark` 位于构建目录下的 `tests/benchmarks/`。在**仓库根目录**执行（`--config` 为相对于当前工作目录的路径，与 `infer_benchmark` 内建示例一致）：

```shell
./build/tests/benchmarks/vision_infer_benchmark \
  --config examples/yolov8/config/yolov8.yaml \
  --image ~/.cache/assets/image/006_test.jpg
```

若 Vision 作为 SDK 子目录且路径仍为 `components/model_zoo/vision/`，可将 `--config` 换为 `components/model_zoo/vision/examples/yolov8/config/yolov8.yaml`。

> 以上命令默认使用 yolov8n_no_dfl 模型；如需指定其他模型，可使用 `--model-path` 参数，例如：`--model-path /path/to/yolov8s_no_dfl.q.onnx`。

**不包含前后处理：**

> 说明：以下数据基于 K1/K3 平台实测（推理引擎 2.0.6），为阶段性信息，持续优化中，请以最新文档为准。

K1:

|  模型大类   |       具体模型        |   输入大小    | 数据类型 | 帧率(4核) |
| :---------: | :-------------------: | :-----------: | :------: | :-------: |
|   resnet    |       resnet50        | [1,3,224,224] |   int8   |   29.9    |
|  mobilenet  |     mobilenet_v1      | [1,3,224,224] |   int8   |   104.1   |
|             |     mobilenet_v2      | [1,3,224,224] |   int8   |   80.6    |
|   arcface   | arcface_mobilefacenet | [1,3,112,112] |   int8   |   124.7   |
| yolov5-face |     yolov5n-face      | [1,3,640,640] |   int8   |    7.6    |
|   yolov8    |        yolov8n        | [1,3,640,640] |   int8   |   14.7    |
|             |        yolov8s        | [1,3,640,640] |   int8   |    6.8    |
|             |        yolov8m        | [1,3,640,640] |   int8   |    3.4    |
| yolov8-pose |     yolov8n-pose      | [1,3,640,640] |   int8   |   11.9    |
|             |     yolov8s-pose      | [1,3,640,640] |   int8   |    6.0    |
|             |     yolov8m-pose      | [1,3,640,640] |   int8   |    3.2    |
| yolov8-seg  |      yolov8n-seg      | [1,3,640,640] |   int8   |   11.5    |
|             |      yolov8s-seg      | [1,3,640,640] |   int8   |    5.4    |
|             |      yolov8m-seg      | [1,3,640,640] |   int8   |    2.7    |
|   yolo11    |        yolo11n        | [1,3,640,640] |   int8   |   12.6    |
|             |        yolo11s        | [1,3,640,640] |   int8   |    6.2    |
|             |        yolo11m        | [1,3,640,640] |   int8   |    2.4    |
|   yolo26    |        yolo26n        | [1,3,640,640] |   int8   |   20.1    |

K3:

|  模型大类   |       具体模型        |   输入大小    | 数据类型 | 帧率(4核) | 帧率(8核) |
| :---------: | :-------------------: | :-----------: | :------: | :-------: | :-------: |
|   resnet    |       resnet18        | [1,3,224,224] |   int8   |   345.7   |   483.1   |
|             |       resnet50        | [1,3,224,224] |   int8   |   130.4   |   186.3   |
|  mobilenet  |     mobilenet_v1      | [1,3,224,224] |   int8   |   257.6   |   430.3   |
|             |     mobilenet_v2      | [1,3,224,224] |   int8   |   193.0   |   308.2   |
|             |  mobilenet_v3_small   | [1,3,224,224] |   fp16   |   311.7   |   367.5   |
|             |  mobilenet_v3_large   | [1,3,224,224] |   fp16   |   168.7   |   223.3   |
|efficientnet |   efficientnet_b0     | [1,3,224,224] |   int8   |   78.2    |   102.2   |
|             |   efficientnet_b1     | [1,3,224,224] |   int8   |   50.6    |   68.0    |
|             |  efficientnet_v2_s    | [1,3,224,224] |   int8   |   50.8    |   74.9    |
|     vit     |        vit_b_16       | [1,3,224,224] |   int8   |   28.5    |   43.1    |
|   arcface   | arcface_mobilefacenet | [1,3,112,112] |   int8   |   280.7   |   434.8   |
|             |      w600k_r50        | [1,3,112,112] |   int8   |   66.0    |   98.4    |
|   adaface   |   adaface_ir101       | [1,3,112,112] |   int8   |   24.1    |   30.2    |
|    scrfd    |     det_10g_fixed     | [1,3,640,640] |   int8   |   48.2    |   79.3    |
|  landmark   |       2d106det        | [1,3,192,192] |   fp32   |   92.9    |   131.4   |
|  genderage  |       genderage       |  [1,3,96,96]  |   int8   |  1239.6   |  1501.7   |
|   emotion   | emotion_resnet50      | [1,3,224,224] |   int8   |   142.4   |   202.7   |
| yolov5-face |     yolov5n-face      | [1,3,640,640] |   int8   |   34.6    |   45.5    |
|   yolov5    |        yolov5n        | [1,3,640,640] |   int8   |   69.4    |   102.8   |
|             |        yolov5s        | [1,3,640,640] |   int8   |   41.3    |   62.2    |
|             |    yolov5_gesture     | [1,3,640,640] |   int8   |   46.6    |   73.9    |
|   yolov8    |        yolov8n        | [1,3,640,640] |   int8   |   70.5    |   106.7   |
|             |        yolov8s        | [1,3,640,640] |   int8   |   39.4    |   59.0    |
|             |        yolov8m        | [1,3,640,640] |   int8   |   20.4    |   31.2    |
| yolov8-pose |     yolov8n-pose      | [1,3,640,640] |   int8   |   61.5    |   88.0    |
|             |     yolov8s-pose      | [1,3,640,640] |   int8   |   35.4    |   52.2    |
|             |     yolov8m-pose      | [1,3,640,640] |   int8   |   19.2    |   29.0    |
| yolov8-seg  |      yolov8n-seg      | [1,3,640,640] |   int8   |   51.4    |   77.3    |
|             |      yolov8s-seg      | [1,3,640,640] |   int8   |   29.9    |   45.3    |
|             |      yolov8m-seg      | [1,3,640,640] |   int8   |   16.2    |   24.9    |
|   yolo11    |        yolo11n        | [1,3,640,640] |   int8   |   50.2    |   76.1    |
|             |        yolo11s        | [1,3,640,640] |   int8   |   31.7    |   48.0    |
|             |        yolo11m        | [1,3,640,640] |   int8   |   14.9    |   23.1    |
|   yolo12    |        yolo12n        | [1,3,640,640] |   int8   |   29.5    |   43.8    |
|             |        yolo12s        | [1,3,640,640] |   int8   |   17.1    |   26.7    |
|             |        yolo12m        | [1,3,640,640] |   int8   |    8.9    |   14.1    |
|   yolo26    |        yolo26n        | [1,3,640,640] |   int8   |   48.8    |   70.1    |
|             |        yolo26s        | [1,3,640,640] |   int8   |   22.0    |   32.6    |
|             |        yolo26m        | [1,3,640,640] |   int8   |   10.7    |   16.2    |
|   ppocr     | PP-OCRv6_tiny_det     | [1,3,640,640] |   fp16   |   17.5    |   27.3    |
|             | PP-OCRv6_tiny_rec     | [1,3,48,320]  |   int8   |   324.4   |   434.6   |
|             | PP-OCRv6_small_det    | [1,3,640,640] |   fp16   |   12.0    |   19.1    |
|             | PP-OCRv6_small_rec    | [1,3,48,320]  |   int8   |   95.9    |   125.5   |


**复现方法**：

参照 2.1 节安装 C++ 依赖后，再按照2.2下载模型，之后可使用 onnxruntime_perf_test 复现（以 yolov8n 为例）：

```shell
onnxruntime_perf_test ~/.cache/models/vision/yolov8/yolov8n_no_dfl.q.onnx  -e spacemit -r 20 -x 1 -S 1 -s -I -c 1 -i "SPACEMIT_EP_INTRA_THREAD_NUM|4"
```

详细说明见 SpacemiT 社区文档 [AI 计算栈 · ONNX Runtime](https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/compute_stack/ai_compute_stack/onnxruntime.md) 中的 **onnxruntime_perf_test** 章节。



