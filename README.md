# Vision 组件

## 1. 项目简介

本组件为 SpacemiT **计算机视觉模型部署库**，基于 ONNX Runtime，在 C++ 与 Python 下提供统一的推理与封装接口，便于集成到机器人、嵌入式与服务器等场景。功能特性如下：

| 类别     | 支持                                                                 |
| -------- | -------------------------------------------------------------------- |
| 模型类型 | 检测、分类、分割、跟踪、人脸、姿态等，一套接口接入                   |
| 后端     | ONNX Runtime（含 SpaceMIT 等定制运行时）                             |
| 接口     | C++（`vision_service.h` 类接口，`cv::Mat` 入参）、Python（ModelFactory / 各模型类） |
| 可扩展   | 抽象基类与公共工具（预处理、NMS、绘图等）便于接入新模型             |

## 2. 验证模型

按以下顺序完成依赖安装、模型准备与示例运行。

### 2.1. 安装依赖

- **编译环境**：CMake 3.10+，C++17；Python 3.12+（可选）。
- **C++ 依赖**：OpenCV 4（core、imgproc、highgui）、spacemit-ort、yaml-cpp。路径可通过 `OpenCV_DIR`、`SPACEMIT_DIR` 等在 CMake 中设置。
- **Python 依赖**：NumPy、OpenCV 4.12+、ONNX Runtime、Pillow、scipy、pydantic、spacemit-ort。

```bash
# 系统依赖示例（Linux）
sudo apt-get update
sudo apt-get install -y python3-spacemit-ort opencv-spacemit
sudo apt-get install  libeigen3-dev

```

**Python 安装（推荐先用于跑通示例）：**
```bash
cd components/model_zoo/vision
pip install -e .
```

### 2.2. 下载模型

模型统一放在 **`~/.cache/models/vision/<type>/`**（如 `~/.cache/models/vision/yolov8/`）。若运行示例时提示「Model file not found」，请在对应示例目录下执行下载脚本：

```bash
# 下载example/yolov8所需要的模型
bash examples/yolov8/scripts/download_models.sh
```

也可以一键下载所有示例/应用所需模型（在 vision 组件根目录执行）：

```bash
bash scripts/download_all_models.sh
```

全部下载完成后模型如下：

```bash
bianbu@k3:~/.cache/models/vision# tree
.
├── arcface
│   └── arcface_mobilefacenet_cut.q.onnx
├── ocsort
│   ├── ocsort_yoloxs_sim.onnx
│   └── yolov8n.q.onnx
├── resnet
│   ├── emotion_resnet50_final.q.onnx
│   └── resnet50.q.onnx
├── stgcn
│   └── stgcn.onnx
├── yolov11
│   ├── yolo11m.q.onnx
│   ├── yolo11n.q.onnx
│   └── yolo11s.q.onnx
├── yolov5
│   └── yolov5_gesture.q.onnx
├── yolov5-face
│   └── yolov5n-face_cut.q.onnx
├── yolov8
│   ├── yolov8_fire.q.onnx
│   ├── yolov8m.q.onnx
│   ├── yolov8n.q.onnx
│   └── yolov8s.q.onnx
├── yolov8_pose
│   ├── yolov8m-pose.q.onnx
│   ├── yolov8n-pose.q.onnx
│   └── yolov8s-pose.q.onnx
└── yolov8_seg
    ├── yolov8m-seg.q.onnx
    ├── yolov8n-seg.q.onnx
    └── yolov8s-seg.q.onnx
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
│   └── 009_test_unet.jpg
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

**编译**：本组件已纳入 SpacemiT Robot SDK 时，在 SDK 根目录下执行。SDK 拉取与初始化见 [SpacemiT Robot SDK Manifest](https://github.com/spacemit-robotics/manifest)（使用 repo 时需先完成 `repo init`、`repo sync` 等）。

```bash
source build/envsetup.sh
cd components/model_zoo/vision
mm
```

构建产物会安装到 `output/staging`。

**运行**：运行前在 SDK 根目录执行 `source build/envsetup.sh`，使 PATH 与库路径指向 `output/staging`，然后可执行：

**Python 示例（以 YOLOv8 为例）：**

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

更多模型（ByteTrack、OC-SORT、ArcFace 等）的用法与参数见**各示例子目录 README**。

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
| OpenCV | `opencv_core`、`opencv_imgproc`、`opencv_highgui`（对应 CMake 中 `find_package(OpenCV COMPONENTS core imgproc highgui)` 与 `${OpenCV_LIBS}`） |
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
| `BUILD_TESTS` | 是否构建测试与 benchmark（默认 OFF） |

### 3.2. API 使用与 CMake 集成

- **C++**：`#include "vision_service.h"`，包含路径指向本组件 `include/`（或安装后的 `include`）；链接 `vision` 及上表所列依赖库。接口为类 `VisionService`，创建与推理、绘制均使用 `cv::Mat`，无需 raw buffer 转换。
- **Python**：通过 **ModelFactory** 从 yaml 创建模型，或直接实例化各模型类并传入 `model_path`、`conf_threshold` 等；各示例的 yaml 与参数见对应子目录 README。

**C++ 调用示例（图像推理 + 绘制）：**

```cpp
#include "vision_service.h"
#include <opencv2/opencv.hpp>

auto service = VisionService::Create(config_path, model_path_override, true);
if (!service) { /* 使用 VisionService::LastCreateError() */ return; }

cv::Mat image = cv::imread(image_path);
std::vector<VisionServiceResult> results;
if (service->InferImage(image, &results) != VISION_SERVICE_OK) { /* 使用 service->LastError() */ return; }

cv::Mat vis;
if (!results.empty())
    service->Draw(image, &vis);
else
    vis = image;
// 使用 vis 显示或保存
```

**CMake 集成示例**：将本组件作为子目录添加后，对目标执行 `target_link_libraries(your_app PRIVATE vision ${OpenCV_LIBS} onnxruntime spacemit_ep yaml-cpp)`，并 `target_include_directories(your_app PRIVATE path/to/vision/include)`。

## 4. 常见问题

1. **Python 导入报错 `No module named 'core'` 等**  
   请在组件根目录执行 `pip install -e .`，或设置 `PYTHONPATH` 包含 `src`。

2. **模型文件未找到**  
   运行 `examples/<model>/scripts/download_models.sh` 将模型下载到 `~/.cache/models/vision/<type>/`，详见 2.2。

3. **依赖版本**  
   Python 以 `setup.py` 中 `install_requires` 为准；C++ 以 CMake 要求及 CI 环境为准；大版本升级若有 ABI/API 变更会在版本与发布中说明。

## 5. 版本与发布

版本以本目录 `CMakeLists.txt` / `setup.py` 为准。

| 版本   | 说明 |
| ------ | ---- |
| 0.1.0  | 计算机视觉模型部署库，支持 YOLOv8、ByteTrack、OC-SORT、ArcFace、ResNet 等；C++ / Python 统一接口，ONNX Runtime 后端。 |

重要变更与兼容性说明将随版本更新在本文档或仓库 Release 中记录。

## 6. 贡献方式

欢迎提交 Issue 与 Pull Request。开发前请确认编码风格与现有 C++/Python 风格一致，并通过相关测试。贡献者与维护者名单见 **`CONTRIBUTORS.md`**（如有）。

## 7. License

本组件源码文件头声明为 Apache-2.0，最终以本目录 **`LICENSE`** 文件为准。

## 8. 附录：模型性能

**包含前后处理：**

> 说明：以下数据基于K1/ K3 平台实测，为阶段性信息，持续优化中，请以最新文档为准。

K1:

|  模型大类   |       具体模型        |   输入大小    | 数据类型 | 帧率(4核) |
| :---------: | :-------------------: | :-----------: | :------: | :-------: |
|   resnet    |       resnet50        | [1,3,224,224] |   int8   |   26.2    |
|   arcface   | arcface_mobilefacenet | [1,3,112,112] |   int8   |   31.3    |
| yolov5-face |     yolov5n-face      | [1,3,640,640] |   int8   |    5.5    |
|   yolov8    |        yolov8n        | [1,3,640,640] |   int8   |   11.8    |
|             |        yolov8s        | [1,3,640,640] |   int8   |    6.6    |
|             |        yolov8m        | [1,3,640,640] |   int8   |    3.4    |
| yolov8-pose |     yolov8n-pose      | [1,3,640,640] |   int8   |   10.0    |
|             |     yolov8s-pose      | [1,3,640,640] |   int8   |    5.9    |
|             |     yolov8m-pose      | [1,3,640,640] |   int8   |    3.2    |
| yolov8-seg  |      yolov8n-seg      | [1,3,640,640] |   int8   |    2.0    |
|             |      yolov8s-seg      | [1,3,640,640] |   int8   |    1.5    |
|             |      yolov8m-seg      | [1,3,640,640] |   int8   |    1.1    |
|   yolo11    |        yolo11n        | [1,3,640,640] |   int8   |    5.4    |
|             |        yolo11s        | [1,3,640,640] |   int8   |    3.0    |
|             |        yolo11m        | [1,3,640,640] |   int8   |    1.2    |

K3:

|  模型大类   |       具体模型        |   输入大小    | 数据类型 | 帧率(4核) | 帧率(8核) |
| :---------: | :-------------------: | :-----------: | :------: | :-------: | :-------: |
|   resnet    |       resnet50        | [1,3,224,224] |   int8   |   108.7   |           |
|   arcface   | arcface_mobilefacenet | [1,3,112,112] |   int8   |    49     |           |
| yolov5-face |     yolov5n-face      | [1,3,640,640] |   int8   |   23.0    |           |
|   yolov8    |        yolov8n        | [1,3,640,640] |   int8   |   37.9    |           |
|             |        yolov8s        | [1,3,640,640] |   int8   |   27.5    |           |
|             |        yolov8m        | [1,3,640,640] |   int8   |   16.8    |           |
| yolov8-pose |     yolov8n-pose      | [1,3,640,640] |   int8   |   39.8    |           |
|             |     yolov8s-pose      | [1,3,640,640] |   int8   |   28.5    |           |
|             |     yolov8m-pose      | [1,3,640,640] |   int8   |   17.2    |           |
| yolov8-seg  |      yolov8n-seg      | [1,3,640,640] |   int8   |    4.7    |           |
|             |      yolov8s-seg      | [1,3,640,640] |   int8   |    3.5    |           |
|             |      yolov8m-seg      | [1,3,640,640] |   int8   |    3.4    |           |
|   yolo11    |        yolo11n        | [1,3,640,640] |   int8   |    9.5    |           |
|             |        yolo11s        | [1,3,640,640] |   int8   |    7.1    |           |
|             |        yolo11m        | [1,3,640,640] |   int8   |    4.0    |           |



**复现方法**：

参照 3.1 节完成 C++ 构建，可执行下面命令测试性能（以 yolov8 为例）：

```shell
./tests/benchmarks/vision_infer_benchmark --config ../examples/yolov8/config/yolov8.yaml --image /path/to/image.jpg
```

> 以上命令默认使用 yolov8n 模型；如需指定其他模型，可使用 `--model-path` 参数，例如：`--model-path /path/to/yolov8s.onnx`。

**不包含前后处理：**

> 说明：以下数据基于 K1/K3 平台实测，为阶段性信息，持续优化中，请以最新文档为准。

K1:

|  模型大类   |       具体模型        |   输入大小    | 数据类型 | 帧率(4核) |
| :---------: | :-------------------: | :-----------: | :------: | :-------: |
|   resnet    |       resnet50        | [1,3,224,224] |   int8   |   30.7    |
|   arcface   | arcface_mobilefacenet | [1,3,112,112] |   int8   |   38.0    |
| yolov5-face |     yolov5n-face      | [1,3,640,640] |   int8   |    6.8    |
|   yolov8    |        yolov8n        | [1,3,640,640] |   int8   |   14.8    |
|             |        yolov8s        | [1,3,640,640] |   int8   |    7.5    |
|             |        yolov8m        | [1,3,640,640] |   int8   |    3.6    |
| yolov8-pose |     yolov8n-pose      | [1,3,640,640] |   int8   |   12.3    |
|             |     yolov8s-pose      | [1,3,640,640] |   int8   |    6.6    |
|             |     yolov8m-pose      | [1,3,640,640] |   int8   |    3.3    |
| yolov8-seg  |      yolov8n-seg      | [1,3,640,640] |   int8   |   11.6    |
|             |      yolov8s-seg      | [1,3,640,640] |   int8   |    5.7    |
|             |      yolov8m-seg      | [1,3,640,640] |   int8   |    2.8    |
|   yolo11    |        yolo11n        | [1,3,640,640] |   int8   |    5.9    |
|             |        yolo11s        | [1,3,640,640] |   int8   |    3.1    |
|             |        yolo11m        | [1,3,640,640] |   int8   |    1.2    |

K3:

|  模型大类   |       具体模型        |   输入大小    | 数据类型 | 帧率(4核) | 帧率(8核) |
| :---------: | :-------------------: | :-----------: | :------: | :-------: | :-------: |
|   resnet    |       resnet50        | [1,3,224,224] |   int8   |   131.9   |           |
|   arcface   | arcface_mobilefacenet | [1,3,112,112] |   int8   |    49     |           |
| yolov5-face |     yolov5n-face      | [1,3,640,640] |   int8   |   28.8    |           |
|   yolov8    |        yolov8n        | [1,3,640,640] |   int8   |   55.6    |           |
|             |        yolov8s        | [1,3,640,640] |   int8   |   27.5    |           |
|             |        yolov8m        | [1,3,640,640] |   int8   |   19.6    |           |
| yolov8-pose |     yolov8n-pose      | [1,3,640,640] |   int8   |   45.3    |           |
|             |     yolov8s-pose      | [1,3,640,640] |   int8   |   30.8    |           |
|             |     yolov8m-pose      | [1,3,640,640] |   int8   |   17.9    |           |
| yolov8-seg  |      yolov8n-seg      | [1,3,640,640] |   int8   |   24.1    |           |
|             |      yolov8s-seg      | [1,3,640,640] |   int8   |   13.9    |           |
|             |      yolov8m-seg      | [1,3,640,640] |   int8   |    8.5    |           |
|   yolo11    |        yolo11n        | [1,3,640,640] |   int8   |    9.5    |           |
|             |        yolo11s        | [1,3,640,640] |   int8   |    7.1    |           |
|             |        yolo11m        | [1,3,640,640] |   int8   |    4.0    |           |

**复现方法**：

参照 2.1 节安装 C++ 依赖后，再按照2.2下载模型，之后可使用 onnxruntime_perf_test 复现（以 yolov8n 为例）：

```shell
onnxruntime_perf_test ~/.cache/models/vision/yolov8/yolov8n.q.onnx  -r 20 -x 4 -S 1 -s -I -c 1 -e spacemit
```

详细参考：https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/compute_stack/ai_compute_stack/onnxruntime.md 中 onnxruntime_perf_test 章节。

