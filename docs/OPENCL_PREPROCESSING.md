# OpenCL Image Preprocessing

VisionService keeps CPU preprocessing as the default. Models that explicitly
opt in to the backend-neutral preprocessing dispatcher can select a backend
with:

```yaml
default_params:
  preprocess:
    backend: opencl
```

The shared dispatcher accepts three policies:

| Policy | Behavior |
| --- | --- |
| `cpu` | Always uses the existing CPU preprocessing path. |
| `auto` | Opted-in models use OpenCL for NV12 DMA-BUF input and CPU for host/BGR input; non-opted-in models remain on CPU. |
| `opencl` | Opted-in models require OpenCL and accept BGR8 host upload or NV12 DMA-BUF input; non-opted-in models reject the configuration. |

`auto` emits one warning when OpenCL is disabled. If execution fails after
external memory has been acquired or a kernel has been enqueued, the current
request fails; only subsequent requests fall back to CPU. Invalid input never
falls back.

Runtime profiling records the implementation actually used as
`image_preprocess.cpu` or `image_preprocess.opencl`, so an `auto` fallback is
visible in benchmark output.

The common preprocessing path is currently enabled for the following
detection, tracking, pose, segmentation, and open-vocabulary models:

- YOLOv5, YOLOv5-Face, YOLOv5-Gesture, and YOLO26;
- YOLOv8, YOLOv11, and YOLOv12 through `YOLOv8Detector`;
- YOLOv8 Pose and YOLOv8 Seg;
- YOLO-World and YOLOE;
- ByteTrack and OCSort through their internal YOLOv8 detector.

SCRFD, PPOCR, PP-LiteSeg, and classification, embedding, face-attribute,
landmark, and other image-text encoder models remain CPU-only in this
delivery. `auto` keeps these models on CPU and strict `opencl` rejects their
configuration.
YOLO-World and YOLOE retain their existing request-local geometry and text or
prompt state; their OpenCL image result remains alive until the synchronous
multi-input ONNX Runtime call completes.

The existing `VisionServiceRequest` fields select pixel format and memory
without introducing separate APIs:

| `image_format` | `image_dma_fd` | OpenCL input |
| --- | ---: | --- |
| `BGR8` | `< 0` | CPU in `auto`; host upload in strict `opencl` |
| `NV12` | `< 0` | CPU in `auto`; rejected by strict `opencl` |
| `BGR8` | `>= 0` | CPU in `auto`; host upload in strict `opencl` (`image_dma_fd` is ignored) |
| `NV12` | `>= 0` | imported DMA-BUF NV12 images |

Strict `opencl` uploads BGR8 pixels from the `cv::Mat` host buffer into a
reusable OpenCL buffer. This path copies the input and completes the upload
before the caller's host storage may be released. `auto` deliberately keeps
BGR8 on CPU. DMA-BUF import is the zero-copy NV12 input path on supported
SpacemiT platforms. The fd, mapped layout, and MPP frame lease must remain
valid for the duration of synchronous `Infer()`.

`MppFrameSource` exposes a move-only native frame lease for camera/video
examples that use it. A compatible MPP NV12 frame remains owned until synchronous
`Infer()` returns, then it may be converted to BGR for drawing and display.
Capture selection (`--use-mpp`) and preprocessing policy remain independent;
the dispatcher only observes pixel format and DMA fd. Split-plane or otherwise
incompatible MPP layouts are converted to owned BGR instead of being exposed as
NV12 DMA input.

The OpenCL backend performs crop, bilinear or nearest resize, letterbox or
top-left padding, channel ordering, per-channel normalization, and NCHW
packing in one fused kernel enqueue after BGR upload or NV12 DMA-BUF import.
It also performs NV12 conversion in the fused NV12 kernels. It supports FP32
and FP16 output; current model preprocess specs request FP32 tensors and batch
size 1. In `auto`, a model whose input tensor has a larger fixed batch stays on
CPU; strict `opencl` rejects it. NV12 conversion follows OpenCV's BT.601
limited-range baseline, and bilinear interpolation preserves the CPU operation
order of conversion before resize. Output tensors use a three-slot DMA-BUF
ring. DMA input imports are retained in a bounded 32-entry cache keyed by
DMA-BUF identity and layout, which prevents fd-number reuse from returning
stale images.

Kernel sources are maintained as `.cl` and `.clh` files under
`src/backends/opencl/kernels`. CMake embeds them into the library at build
time, so deployment does not depend on kernel source paths.

OpenCL support is optional and defaults to off:

```bash
cmake -S . -B build -DVISION_WITH_OPENCL=ON
cmake --build build -j
```

With `VISION_WITH_OPENCL=OFF`, CMake does not discover or link OpenCL.
`auto` remains on CPU, while explicit `opencl` is rejected during
configuration. The OpenCL implementation is isolated under
`src/backends/opencl/{runtime,memory,operators,kernels}` and built as the
independent `vision_opencl_backend` target.

MPP capture and preprocessing backend selection remain independent:
`--use-mpp` chooses the input transport, while `preprocess.backend` chooses
how an opted-in model consumes that input.
