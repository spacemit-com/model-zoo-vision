# OpenCL Image Preprocessing

VisionService keeps CPU preprocessing as the default. To select the OpenCL
backend for an image model, add:

```yaml
default_params:
  preprocess:
    backend: opencl
```

The existing `VisionServiceRequest` fields select pixel format and memory
without introducing separate APIs:

| `image_format` | `image_dma_fd` | OpenCL input |
| --- | ---: | --- |
| `BGR8` | `< 0` | `CL_MEM_USE_HOST_PTR` BGR |
| `NV12` | `< 0` | `CL_MEM_USE_HOST_PTR` NV12 |
| `BGR8` | `>= 0` | imported DMA-BUF BGR |
| `NV12` | `>= 0` | imported DMA-BUF NV12 images |

`CL_MEM_USE_HOST_PTR` is a host-memory interoperability mode, not a zero-copy
guarantee. The driver may copy the input. DMA-BUF import is the zero-copy input
path on supported SpacemiT platforms.

For host inputs, the caller owns the `cv::Mat` storage and must not write to it
while `Infer()` is running. Inference is synchronous: the OpenCL queue has
finished reading the host pointer before `Infer()` returns. For DMA inputs,
the fd and its mapped layout must remain valid for the duration of `Infer()`.

The common backend performs crop, bilinear or nearest resize, letterbox or
top-left padding, BGR/NV12 conversion, channel ordering, per-channel
normalization, and NCHW packing in one kernel. It supports FP32 and FP16 output;
the current ONNX image models request FP32 tensors. Output tensors use a
three-slot DMA-BUF ring. DMA input imports are retained in a bounded 32-entry
cache keyed by DMA-BUF identity and layout, which prevents fd-number reuse from
returning stale images.

Explicit `backend: opencl` never silently falls back to CPU. Missing OpenCL,
DMA-BUF, image-view, or FP16 capabilities are reported as inference errors.
Omitting `preprocess.backend`, or setting it to `cpu`, preserves the existing
CPU preprocessing behavior.

Direct image model classes are supported. Sequence-only ST-GCN and
Emotion-LSTM do not have an image preprocessing stage. PP-OCR moves its DBNet
input preprocessing to OpenCL; result-driven perspective crops and recognition
of those dynamic small images remain on CPU.
