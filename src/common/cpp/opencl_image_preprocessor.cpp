/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "opencl_image_preprocessor.h"

#include <CL/cl_ext.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace vision_common {
namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kInputCacheCapacity = 32;

const char* kKernelSource = R"CLC(
__constant sampler_t nv12_sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

inline uchar read_u8(__global const uchar* input, int offset)
{
    return input[offset];
}

inline float read_bgr_channel(
    __global const uchar* input,
    int stride,
    int width,
    int height,
    int x,
    int y,
    int channel)
{
    x = clamp(x, 0, width - 1);
    y = clamp(y, 0, height - 1);
    return convert_float(read_u8(input, y * stride + x * 3 + channel));
}

inline float sample_bgr_channel(
    __global const uchar* input,
    int stride,
    int width,
    int height,
    float x,
    float y,
    int channel)
{
    int x0 = convert_int_rtn(floor(x));
    int y0 = convert_int_rtn(floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = x - floor(x);
    float ay = y - floor(y);
    float p00 = read_bgr_channel(input, stride, width, height, x0, y0, channel);
    float p01 = read_bgr_channel(input, stride, width, height, x1, y0, channel);
    float p10 = read_bgr_channel(input, stride, width, height, x0, y1, channel);
    float p11 = read_bgr_channel(input, stride, width, height, x1, y1, channel);
    return mix(mix(p00, p01, ax), mix(p10, p11, ax), ay);
}

inline float3 sample_bgr(
    __global const uchar* input,
    int stride,
    int width,
    int height,
    float x,
    float y)
{
    return (float3)(
        sample_bgr_channel(input, stride, width, height, x, y, 0),
        sample_bgr_channel(input, stride, width, height, x, y, 1),
        sample_bgr_channel(input, stride, width, height, x, y, 2));
}

inline float sample_plane(
    __global const uchar* input,
    int offset,
    int stride,
    int width,
    int height,
    float x,
    float y,
    int channel,
    int channels)
{
    int x0 = convert_int_rtn(floor(x));
    int y0 = convert_int_rtn(floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = x - floor(x);
    float ay = y - floor(y);
    x0 = clamp(x0, 0, width - 1);
    x1 = clamp(x1, 0, width - 1);
    y0 = clamp(y0, 0, height - 1);
    y1 = clamp(y1, 0, height - 1);
    float p00 = convert_float(read_u8(
        input, offset + y0 * stride + x0 * channels + channel));
    float p01 = convert_float(read_u8(
        input, offset + y0 * stride + x1 * channels + channel));
    float p10 = convert_float(read_u8(
        input, offset + y1 * stride + x0 * channels + channel));
    float p11 = convert_float(read_u8(
        input, offset + y1 * stride + x1 * channels + channel));
    return mix(mix(p00, p01, ax), mix(p10, p11, ax), ay);
}

inline float3 yuv_to_rgb(float y, float u, float v)
{
    float c = y - 16.0f;
    float d = u - 128.0f;
    float e = v - 128.0f;
    return clamp(
        (float3)(
            1.16438356f * c + 1.79274107f * e,
            1.16438356f * c - 0.21324861f * d - 0.53290933f * e,
            1.16438356f * c + 2.11240179f * d),
        0.0f, 255.0f);
}

inline float3 sample_nv12_buffer(
    __global const uchar* input,
    int y_stride,
    int uv_stride,
    int uv_offset,
    int width,
    int height,
    float x,
    float y)
{
    float yy = sample_plane(
        input, 0, y_stride, width, height, x, y, 0, 1);
    float u = sample_plane(
        input, uv_offset, uv_stride, width / 2, height / 2,
        x * 0.5f, y * 0.5f, 0, 2);
    float v = sample_plane(
        input, uv_offset, uv_stride, width / 2, height / 2,
        x * 0.5f, y * 0.5f, 1, 2);
    return yuv_to_rgb(yy, u, v);
}

inline float3 sample_nv12_images(
    read_only image2d_t y_image,
    read_only image2d_t uv_image,
    float x,
    float y)
{
    float yy = read_imagef(
        y_image, nv12_sampler, (float2)(x + 0.5f, y + 0.5f)).x * 255.0f;
    float2 uv = read_imagef(
        uv_image, nv12_sampler,
        (float2)(x * 0.5f + 0.5f, y * 0.5f + 0.5f)).xy * 255.0f;
    return yuv_to_rgb(yy, uv.x, uv.y);
}

inline float2 source_coordinate(
    int x,
    int y,
    int dst_x,
    int dst_y,
    int dst_width,
    int dst_height,
    float src_x,
    float src_y,
    float src_width,
    float src_height)
{
    return (float2)(
        src_x + ((float)(x - dst_x) + 0.5f) * src_width /
            (float)dst_width - 0.5f,
        src_y + ((float)(y - dst_y) + 0.5f) * src_height /
            (float)dst_height - 0.5f);
}

inline float3 reorder_normalize(
    float3 bgr,
    int output_rgb,
    float4 mean,
    float4 scale)
{
    float3 value = output_rgb ? bgr.zyx : bgr;
    return (value - mean.xyz) * scale.xyz;
}

inline void store_float_output(
    __global float* output,
    int index,
    int plane,
    float3 value)
{
    output[index] = value.x;
    output[index + plane] = value.y;
    output[index + plane * 2] = value.z;
}

inline void store_half_output(
    __global half* output,
    int index,
    int plane,
    float3 value)
{
    vstore_half(value.x, 0, output + index);
    vstore_half(value.y, 0, output + index + plane);
    vstore_half(value.z, 0, output + index + plane * 2);
}

#define COMMON_ARGUMENTS                                                   \
    int input_width, int input_height, int output_width, int output_height,\
    float src_x, float src_y, float src_width, float src_height,           \
    int dst_x, int dst_y, int dst_width, int dst_height,                   \
    int output_rgb, int interpolation,                                  \
    float4 mean, float4 scale, float4 padding

#define PROCESS_PIXEL(SAMPLE_EXPRESSION, OUTPUT_TYPE, STORE_FUNCTION)       \
    int x = get_global_id(0);                                               \
    int y = get_global_id(1);                                               \
    if (x >= output_width || y >= output_height) return;                    \
    float3 value = padding.xyz;                                             \
    if (x >= dst_x && x < dst_x + dst_width &&                              \
        y >= dst_y && y < dst_y + dst_height) {                             \
        float2 source = source_coordinate(                                  \
            x, y, dst_x, dst_y, dst_width, dst_height,                      \
            src_x, src_y, src_width, src_height);                           \
        if (interpolation == 1) source = floor(source + 0.5f);               \
        value = SAMPLE_EXPRESSION;                                          \
        value = output_rgb ? value.zyx : value;                             \
    }                                                                       \
    value = (value - mean.xyz) * scale.xyz;                                 \
    int index = y * output_width + x;                                       \
    int plane = output_width * output_height;                               \
    STORE_FUNCTION(output, index, plane, value)

__kernel void preprocess_bgr_buffer_f32(
    __global const uchar* input,
    int input_stride,
    __global float* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_bgr(input, input_stride, input_width, input_height,
                    source.x, source.y),
        float,
        store_float_output);
}

__kernel void preprocess_nv12_buffer_f32(
    __global const uchar* input,
    int y_stride,
    int uv_stride,
    int uv_offset,
    __global float* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_nv12_buffer(input, y_stride, uv_stride, uv_offset,
                            input_width, input_height, source.x, source.y),
        float,
        store_float_output);
}

__kernel void preprocess_nv12_images_f32(
    read_only image2d_t y_image,
    read_only image2d_t uv_image,
    __global float* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_nv12_images(y_image, uv_image, source.x, source.y),
        float,
        store_float_output);
}

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void preprocess_bgr_buffer_f16(
    __global const uchar* input,
    int input_stride,
    __global half* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_bgr(input, input_stride, input_width, input_height,
                    source.x, source.y),
        half,
        store_half_output);
}

__kernel void preprocess_nv12_buffer_f16(
    __global const uchar* input,
    int y_stride,
    int uv_stride,
    int uv_offset,
    __global half* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_nv12_buffer(input, y_stride, uv_stride, uv_offset,
                            input_width, input_height, source.x, source.y),
        half,
        store_half_output);
}

__kernel void preprocess_nv12_images_f16(
    read_only image2d_t y_image,
    read_only image2d_t uv_image,
    __global half* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_nv12_images(y_image, uv_image, source.x, source.y),
        half,
        store_half_output);
}
#endif
)CLC";

using ImportMemoryArm = cl_mem(CL_API_CALL*)(
    cl_context,
    cl_mem_flags,
    const cl_import_properties_arm*,
    void*,
    size_t,
    cl_int*);
using AcquireExternalMemory = cl_int(CL_API_CALL*)(
    cl_command_queue,
    cl_uint,
    const cl_mem*,
    cl_uint,
    const cl_event*,
    cl_event*);
using ReleaseExternalMemory = AcquireExternalMemory;

struct GeometryPlan {
    float src_x = 0.0F;
    float src_y = 0.0F;
    float src_width = 0.0F;
    float src_height = 0.0F;
    int dst_x = 0;
    int dst_y = 0;
    int dst_width = 0;
    int dst_height = 0;
};

size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

void check_cl(cl_int error, const char* operation)
{
    if (error != CL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            std::to_string(error));
    }
}

void dma_sync(int fd, uint64_t flags)
{
    dma_buf_sync sync{flags};
    if (::ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        throw std::runtime_error(
            "DMA_BUF_IOCTL_SYNC failed: " +
            std::string(std::strerror(errno)));
    }
}

class ScopedClMemory {
public:
    ScopedClMemory() = default;
    explicit ScopedClMemory(cl_mem value) : value_(value) {}
    ~ScopedClMemory()
    {
        if (value_) clReleaseMemObject(value_);
    }

    ScopedClMemory(const ScopedClMemory&) = delete;
    ScopedClMemory& operator=(const ScopedClMemory&) = delete;

    ScopedClMemory(ScopedClMemory&& other) noexcept
        : value_(other.value_)
    {
        other.value_ = nullptr;
    }

    ScopedClMemory& operator=(ScopedClMemory&& other) noexcept
    {
        if (this != &other) {
            if (value_) clReleaseMemObject(value_);
            value_ = other.value_;
            other.value_ = nullptr;
        }
        return *this;
    }

    cl_mem get() const { return value_; }

    void reset()
    {
        if (value_) clReleaseMemObject(value_);
        value_ = nullptr;
    }

private:
    cl_mem value_{nullptr};
};

class MappedDmaBuffer {
public:
    explicit MappedDmaBuffer(size_t size)
        : size_(size), map_size_(align_up(size, kPageSize))
    {
        const char* heaps[] = {
            "/dev/dma_heap/linux,cma",
            "/dev/dma_heap/system"};
        for (const char* path : heaps) {
            const int heap = ::open(path, O_RDWR | O_CLOEXEC);
            if (heap < 0) continue;
            dma_heap_allocation_data allocation{};
            allocation.len = map_size_;
            allocation.fd_flags = O_RDWR | O_CLOEXEC;
            const int result =
                ::ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation);
            ::close(heap);
            if (result == 0) {
                fd_ = allocation.fd;
                break;
            }
        }
        if (fd_ < 0) {
            throw std::runtime_error("failed to allocate output dma-buf");
        }
        data_ = ::mmap(
            nullptr, map_size_, PROT_READ | PROT_WRITE,
            MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to mmap output dma-buf");
        }
    }

    ~MappedDmaBuffer()
    {
        if (data_) ::munmap(data_, map_size_);
        if (fd_ >= 0) ::close(fd_);
    }

    int fd() const { return fd_; }
    void* data() const { return data_; }
    size_t size() const { return size_; }

private:
    int fd_{-1};
    void* data_{nullptr};
    size_t size_{0};
    size_t map_size_{0};
};

GeometryPlan make_geometry(
    const OpenClPreprocessSpec& spec,
    int input_width,
    int input_height)
{
    GeometryPlan plan;
    plan.src_width = static_cast<float>(input_width);
    plan.src_height = static_cast<float>(input_height);
    plan.dst_width = spec.output_width;
    plan.dst_height = spec.output_height;

    if (spec.crop_mode == PreprocessCropMode::kCenterSquare) {
        const float side = static_cast<float>(
            std::min(input_width, input_height));
        plan.src_x = (input_width - side) * 0.5F;
        plan.src_y = (input_height - side) * 0.5F;
        plan.src_width = side;
        plan.src_height = side;
    } else if (
        spec.crop_mode ==
        PreprocessCropMode::kResizeShortSideCenterCrop) {
        if (spec.resize_width <= 0 || spec.resize_height <= 0) {
            throw std::runtime_error(
                "center-crop preprocessing requires resize dimensions");
        }
        const float virtual_scale_x =
            static_cast<float>(spec.resize_width) / input_width;
        const float virtual_scale_y =
            static_cast<float>(spec.resize_height) / input_height;
        plan.src_width =
            static_cast<float>(spec.output_width) / virtual_scale_x;
        plan.src_height =
            static_cast<float>(spec.output_height) / virtual_scale_y;
        plan.src_x = (input_width - plan.src_width) * 0.5F;
        plan.src_y = (input_height - plan.src_height) * 0.5F;
    }

    if (spec.resize_mode != PreprocessResizeMode::kStretch) {
        const float scale = std::min(
            static_cast<float>(spec.output_width) / plan.src_width,
            static_cast<float>(spec.output_height) / plan.src_height);
        plan.dst_width = std::max(
            1, static_cast<int>(std::round(plan.src_width * scale)));
        plan.dst_height = std::max(
            1, static_cast<int>(std::round(plan.src_height * scale)));
        if (spec.resize_mode == PreprocessResizeMode::kLetterbox) {
            plan.dst_x = static_cast<int>(std::round(
                (spec.output_width - plan.dst_width) / 2.0F - 0.1F));
            plan.dst_y = static_cast<int>(std::round(
                (spec.output_height - plan.dst_height) / 2.0F - 0.1F));
        }
    }
    return plan;
}

cl_float4 make_float4(const std::array<float, 3>& values)
{
    cl_float4 result{};
    result.s[0] = values[0];
    result.s[1] = values[1];
    result.s[2] = values[2];
    return result;
}

}  // namespace

class OpenClImagePreprocessor::Impl {
public:
    struct OutputSlot {
        std::unique_ptr<MappedDmaBuffer> buffer;
        ScopedClMemory memory;
        bool cpu_read_active = false;
    };

    struct DmaInput {
        int retained_fd = -1;
        dev_t device = 0;
        ino_t inode = 0;
        vision_core::ImagePixelFormat format =
            vision_core::ImagePixelFormat::kBgr8;
        int width = 0;
        int height = 0;
        int stride = 0;
        size_t total_size = 0;
        ScopedClMemory buffer;
        ScopedClMemory y_image;
        ScopedClMemory uv_sub_buffer;
        ScopedClMemory uv_image;

        ~DmaInput()
        {
            uv_image.reset();
            uv_sub_buffer.reset();
            y_image.reset();
            buffer.reset();
            if (retained_fd >= 0) ::close(retained_fd);
        }
    };

    Impl(const OpenClPreprocessSpec& spec, int output_ring_depth)
        : spec_(spec)
    {
        validate_spec(output_ring_depth);
        initialize_opencl();
        initialize_kernels();
        initialize_output_ring(output_ring_depth);
    }

    ~Impl()
    {
        try {
            finish_cpu_read();
        } catch (...) {
        }
        input_cache_.clear();
        output_ring_.clear();
        for (cl_kernel kernel : kernels_) {
            if (kernel) clReleaseKernel(kernel);
        }
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

    cv::Mat process(const vision_core::ImageInput& input)
    {
        validate_input(input);
        finish_cpu_read();

        output_index_ = (output_index_ + 1U) % output_ring_.size();
        OutputSlot& output = output_ring_[output_index_];
        const int input_width = input.image.cols;
        const int input_height =
            input.format == vision_core::ImagePixelFormat::kNv12
                ? input.image.rows * 2 / 3
                : input.image.rows;
        const GeometryPlan geometry =
            make_geometry(spec_, input_width, input_height);

        cv::Mat retained_host_image;
        ScopedClMemory host_memory;
        DmaInput* dma_input = nullptr;
        cl_mem input_memory = nullptr;
        if (input.dma_fd >= 0) {
            dma_input = &get_dma_input(input);
            input_memory = dma_input->buffer.get();
        } else {
            retained_host_image = input.image;
            cl_int error = CL_SUCCESS;
            input_memory = clCreateBuffer(
                context_,
                CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                required_input_size(input),
                retained_host_image.data,
                &error);
            check_cl(error, "clCreateBuffer(CL_MEM_USE_HOST_PTR)");
            host_memory = ScopedClMemory(input_memory);
        }

        std::vector<cl_mem> external;
        if (dma_input) external.push_back(input_memory);
        external.push_back(output.memory.get());
        check_cl(
            acquire_(
                queue_, static_cast<cl_uint>(external.size()),
                external.data(), 0, nullptr, nullptr),
            "acquire external memory");

        try {
            enqueue(
                input, input_memory, dma_input, output.memory.get(),
                input_width, input_height, geometry);
            check_cl(
                release_(
                    queue_, static_cast<cl_uint>(external.size()),
                    external.data(), 0, nullptr, nullptr),
                "release external memory");
            check_cl(clFinish(queue_), "clFinish");
        } catch (...) {
            release_(
                queue_, static_cast<cl_uint>(external.size()),
                external.data(), 0, nullptr, nullptr);
            clFinish(queue_);
            throw;
        }

        dma_sync(
            output.buffer->fd(),
            DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
        output.cpu_read_active = true;
        active_output_index_ = output_index_;

        const int shape[] = {
            1, 3, spec_.output_height, spec_.output_width};
        const int type =
            spec_.output_type == PreprocessOutputType::kFloat32
                ? CV_32F
                : CV_16F;
        return cv::Mat(
            4, shape, type, output.buffer->data());
    }

    void finish_cpu_read()
    {
        if (active_output_index_ >= output_ring_.size()) return;
        OutputSlot& output = output_ring_[active_output_index_];
        if (output.cpu_read_active) {
            dma_sync(
                output.buffer->fd(),
                DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
            output.cpu_read_active = false;
        }
        active_output_index_ = output_ring_.size();
    }

private:
    enum KernelIndex {
        kBgrFloat32,
        kNv12BufferFloat32,
        kNv12ImagesFloat32,
        kBgrFloat16,
        kNv12BufferFloat16,
        kNv12ImagesFloat16,
        kKernelCount,
    };

    void validate_spec(int output_ring_depth) const
    {
        if (spec_.output_width <= 0 || spec_.output_height <= 0) {
            throw std::runtime_error(
                "OpenCL preprocess output dimensions must be positive");
        }
        if (output_ring_depth < 2 || output_ring_depth > 16) {
            throw std::runtime_error(
                "OpenCL preprocess output ring depth must be in [2, 16]");
        }
    }

    void initialize_opencl()
    {
        cl_int error = clGetPlatformIDs(1, &platform_, nullptr);
        check_cl(error, "clGetPlatformIDs");
        error = clGetDeviceIDs(
            platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
        check_cl(error, "clGetDeviceIDs");
        context_ =
            clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &error);
        check_cl(error, "clCreateContext");
        const cl_queue_properties properties[] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue_ = clCreateCommandQueueWithProperties(
            context_, device_, properties, &error);
        check_cl(error, "clCreateCommandQueueWithProperties");

        import_memory_ = reinterpret_cast<ImportMemoryArm>(
            clGetExtensionFunctionAddressForPlatform(
                platform_, "clImportMemoryARM"));
        acquire_ = reinterpret_cast<AcquireExternalMemory>(
            clGetExtensionFunctionAddressForPlatform(
                platform_, "clEnqueueAcquireExternalMemObjectsKHR"));
        release_ = reinterpret_cast<ReleaseExternalMemory>(
            clGetExtensionFunctionAddressForPlatform(
                platform_, "clEnqueueReleaseExternalMemObjectsKHR"));
        if (!import_memory_ || !acquire_ || !release_) {
            throw std::runtime_error(
                "OpenCL DMA-BUF import/acquire/release is unavailable");
        }
    }

    void initialize_kernels()
    {
        cl_int error = CL_SUCCESS;
        const size_t source_size = std::strlen(kKernelSource);
        program_ = clCreateProgramWithSource(
            context_, 1, &kKernelSource, &source_size, &error);
        check_cl(error, "clCreateProgramWithSource");
        error = clBuildProgram(
            program_, 1, &device_, "-cl-fast-relaxed-math",
            nullptr, nullptr);
        if (error != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(
                program_, device_, CL_PROGRAM_BUILD_LOG,
                0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(
                program_, device_, CL_PROGRAM_BUILD_LOG,
                log.size(), log.data(), nullptr);
            throw std::runtime_error(
                "OpenCL image preprocess build failed: " + log);
        }

        const char* names[] = {
            "preprocess_bgr_buffer_f32",
            "preprocess_nv12_buffer_f32",
            "preprocess_nv12_images_f32",
            "preprocess_bgr_buffer_f16",
            "preprocess_nv12_buffer_f16",
            "preprocess_nv12_images_f16"};
        for (int i = 0; i < kKernelCount; ++i) {
            kernels_[i] = clCreateKernel(program_, names[i], &error);
            if (error != CL_SUCCESS) {
                if (i >= kBgrFloat16 &&
                    spec_.output_type == PreprocessOutputType::kFloat32) {
                    kernels_[i] = nullptr;
                    continue;
                }
                check_cl(error, "clCreateKernel");
            }
        }
    }

    void initialize_output_ring(int depth)
    {
        const size_t element_size =
            spec_.output_type == PreprocessOutputType::kFloat32
                ? sizeof(float)
                : sizeof(uint16_t);
        const size_t output_size =
            static_cast<size_t>(spec_.output_width) *
            spec_.output_height * 3U * element_size;
        output_ring_.reserve(static_cast<size_t>(depth));
        for (int i = 0; i < depth; ++i) {
            OutputSlot slot;
            slot.buffer =
                std::make_unique<MappedDmaBuffer>(output_size);
            slot.memory = ScopedClMemory(
                import_dmabuf(slot.buffer->fd(), slot.buffer->size()));
            output_ring_.push_back(std::move(slot));
        }
        output_index_ = output_ring_.size() - 1U;
        active_output_index_ = output_ring_.size();
    }

    void validate_input(const vision_core::ImageInput& input) const
    {
        if (input.image.empty()) {
            throw std::runtime_error("OpenCL preprocess input is empty");
        }
        if (input.format == vision_core::ImagePixelFormat::kBgr8) {
            if (input.image.type() != CV_8UC3) {
                throw std::runtime_error(
                    "BGR8 OpenCL input must have type CV_8UC3");
            }
        } else if (
            input.image.type() != CV_8UC1 ||
            input.image.rows % 3 != 0 ||
            (input.image.cols & 1) != 0) {
            throw std::runtime_error(
                "NV12 OpenCL input must be CV_8UC1 H*3/2 x W");
        }
        if (!input.image.isContinuous() && input.image.step[0] == 0) {
            throw std::runtime_error(
                "OpenCL input has an invalid row stride");
        }
        if (input.dma_fd >= 0) {
            struct stat info {};
            if (::fstat(input.dma_fd, &info) != 0) {
                throw std::runtime_error(
                    "invalid input dma-buf fd: " +
                    std::string(std::strerror(errno)));
            }
        }
    }

    size_t required_input_size(
        const vision_core::ImageInput& input) const
    {
        return static_cast<size_t>(input.image.step[0]) *
            input.image.rows;
    }

    cl_mem import_dmabuf(int fd, size_t size)
    {
        const cl_import_properties_arm properties[] = {
            CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0};
        cl_int error = CL_SUCCESS;
        cl_mem memory = import_memory_(
            context_, CL_MEM_READ_WRITE, properties,
            &fd, size, &error);
        check_cl(error, "clImportMemoryARM");
        return memory;
    }

    DmaInput& get_dma_input(const vision_core::ImageInput& input)
    {
        struct stat info {};
        if (::fstat(input.dma_fd, &info) != 0) {
            throw std::runtime_error(
                "fstat(input dma-buf) failed: " +
                std::string(std::strerror(errno)));
        }
        const int width = input.image.cols;
        const int height =
            input.format == vision_core::ImagePixelFormat::kNv12
                ? input.image.rows * 2 / 3
                : input.image.rows;
        const int stride = static_cast<int>(input.image.step[0]);
        const size_t total_size = required_input_size(input);

        for (auto iterator = input_cache_.begin();
            iterator != input_cache_.end(); ++iterator) {
            DmaInput& cached = **iterator;
            if (cached.device == info.st_dev &&
                cached.inode == info.st_ino &&
                cached.format == input.format &&
                cached.width == width &&
                cached.height == height &&
                cached.stride == stride &&
                cached.total_size == total_size) {
                if (iterator != input_cache_.begin()) {
                    auto value = std::move(*iterator);
                    input_cache_.erase(iterator);
                    input_cache_.push_front(std::move(value));
                }
                return *input_cache_.front();
            }
        }

        auto cached = std::make_unique<DmaInput>();
        cached->retained_fd = ::dup(input.dma_fd);
        if (cached->retained_fd < 0) {
            throw std::runtime_error(
                "dup(input dma-buf) failed: " +
                std::string(std::strerror(errno)));
        }
        cached->device = info.st_dev;
        cached->inode = info.st_ino;
        cached->format = input.format;
        cached->width = width;
        cached->height = height;
        cached->stride = stride;
        cached->total_size = total_size;
        cached->buffer = ScopedClMemory(
            import_dmabuf(cached->retained_fd, total_size));

        if (input.format == vision_core::ImagePixelFormat::kNv12) {
            create_nv12_images(*cached);
        }
        input_cache_.push_front(std::move(cached));
        if (input_cache_.size() > kInputCacheCapacity) {
            input_cache_.pop_back();
        }
        return *input_cache_.front();
    }

    void create_nv12_images(DmaInput& input)
    {
        cl_image_format y_format{};
        y_format.image_channel_order = CL_R;
        y_format.image_channel_data_type = CL_UNORM_INT8;
        cl_image_desc y_description{};
        y_description.image_type = CL_MEM_OBJECT_IMAGE2D;
        y_description.image_width =
            static_cast<size_t>(input.width);
        y_description.image_height =
            static_cast<size_t>(input.height);
        y_description.image_row_pitch =
            static_cast<size_t>(input.stride);
        y_description.buffer = input.buffer.get();
        cl_int error = CL_SUCCESS;
        input.y_image = ScopedClMemory(clCreateImage(
            context_, CL_MEM_READ_ONLY,
            &y_format, &y_description, nullptr, &error));
        check_cl(error, "clCreateImage(Y)");

        const size_t uv_offset =
            static_cast<size_t>(input.stride) * input.height;
        cl_buffer_region uv_region{};
        uv_region.origin = uv_offset;
        uv_region.size = input.total_size - uv_offset;
        input.uv_sub_buffer = ScopedClMemory(clCreateSubBuffer(
            input.buffer.get(), CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION, &uv_region, &error));
        check_cl(error, "clCreateSubBuffer(UV)");

        cl_image_format uv_format{};
        uv_format.image_channel_order = CL_RG;
        uv_format.image_channel_data_type = CL_UNORM_INT8;
        cl_image_desc uv_description{};
        uv_description.image_type = CL_MEM_OBJECT_IMAGE2D;
        uv_description.image_width =
            static_cast<size_t>(input.width / 2);
        uv_description.image_height =
            static_cast<size_t>(input.height / 2);
        uv_description.image_row_pitch =
            static_cast<size_t>(input.stride);
        uv_description.buffer = input.uv_sub_buffer.get();
        input.uv_image = ScopedClMemory(clCreateImage(
            context_, CL_MEM_READ_ONLY,
            &uv_format, &uv_description, nullptr, &error));
        check_cl(error, "clCreateImage(UV)");
    }

    cl_kernel select_kernel(
        vision_core::ImagePixelFormat format,
        bool use_images) const
    {
        const bool fp16 =
            spec_.output_type == PreprocessOutputType::kFloat16;
        if (format == vision_core::ImagePixelFormat::kBgr8) {
            return kernels_[fp16 ? kBgrFloat16 : kBgrFloat32];
        }
        if (use_images) {
            return kernels_[
                fp16 ? kNv12ImagesFloat16 : kNv12ImagesFloat32];
        }
        return kernels_[
            fp16 ? kNv12BufferFloat16 : kNv12BufferFloat32];
    }

    void enqueue(
        const vision_core::ImageInput& input,
        cl_mem input_memory,
        DmaInput* dma_input,
        cl_mem output_memory,
        int input_width,
        int input_height,
        const GeometryPlan& geometry)
    {
        const bool use_nv12_images =
            input.format == vision_core::ImagePixelFormat::kNv12 &&
            dma_input != nullptr &&
            dma_input->y_image.get() != nullptr &&
            dma_input->uv_image.get() != nullptr;
        cl_kernel kernel =
            select_kernel(input.format, use_nv12_images);
        if (!kernel) {
            throw std::runtime_error(
                "requested OpenCL output type is unsupported");
        }

        cl_int error = CL_SUCCESS;
        int argument = 0;
        if (use_nv12_images) {
            cl_mem y_image = dma_input->y_image.get();
            cl_mem uv_image = dma_input->uv_image.get();
            error |= clSetKernelArg(
                kernel, argument++, sizeof(cl_mem), &y_image);
            error |= clSetKernelArg(
                kernel, argument++, sizeof(cl_mem), &uv_image);
        } else {
            error |= clSetKernelArg(
                kernel, argument++, sizeof(cl_mem), &input_memory);
            const int stride =
                static_cast<int>(input.image.step[0]);
            error |= clSetKernelArg(
                kernel, argument++, sizeof(int), &stride);
            if (input.format == vision_core::ImagePixelFormat::kNv12) {
                const int uv_stride = stride;
                const int uv_offset = stride * input_height;
                error |= clSetKernelArg(
                    kernel, argument++, sizeof(int), &uv_stride);
                error |= clSetKernelArg(
                    kernel, argument++, sizeof(int), &uv_offset);
            }
        }
        error |= clSetKernelArg(
            kernel, argument++, sizeof(cl_mem), &output_memory);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &input_width);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &input_height);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &spec_.output_width);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &spec_.output_height);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(float), &geometry.src_x);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(float), &geometry.src_y);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(float), &geometry.src_width);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(float), &geometry.src_height);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &geometry.dst_x);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &geometry.dst_y);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &geometry.dst_width);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &geometry.dst_height);
        const int output_rgb = spec_.output_rgb ? 1 : 0;
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &output_rgb);
        const int interpolation =
            spec_.interpolation ==
                PreprocessInterpolation::kNearest
            ? 1
            : 0;
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &interpolation);
        const cl_float4 mean = make_float4(spec_.mean);
        const cl_float4 scale = make_float4(spec_.scale);
        const cl_float4 padding = make_float4(spec_.padding);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(cl_float4), &mean);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(cl_float4), &scale);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(cl_float4), &padding);
        check_cl(error, "clSetKernelArg");

        const size_t global[] = {
            static_cast<size_t>(spec_.output_width),
            static_cast<size_t>(spec_.output_height)};
        check_cl(
            clEnqueueNDRangeKernel(
                queue_, kernel, 2, nullptr, global, nullptr,
                0, nullptr, nullptr),
            "clEnqueueNDRangeKernel");
    }

    OpenClPreprocessSpec spec_;
    std::vector<OutputSlot> output_ring_;
    size_t output_index_{0};
    size_t active_output_index_{0};
    std::deque<std::unique_ptr<DmaInput>> input_cache_;
    cl_platform_id platform_{nullptr};
    cl_device_id device_{nullptr};
    cl_context context_{nullptr};
    cl_command_queue queue_{nullptr};
    cl_program program_{nullptr};
    cl_kernel kernels_[kKernelCount]{};
    ImportMemoryArm import_memory_{nullptr};
    AcquireExternalMemory acquire_{nullptr};
    ReleaseExternalMemory release_{nullptr};
};

OpenClImagePreprocessor::OpenClImagePreprocessor(
    const OpenClPreprocessSpec& spec, int output_ring_depth)
    : impl_(std::make_unique<Impl>(spec, output_ring_depth))
{
}

OpenClImagePreprocessor::~OpenClImagePreprocessor() = default;

cv::Mat OpenClImagePreprocessor::process(
    const vision_core::ImageInput& input)
{
    return impl_->process(input);
}

void OpenClImagePreprocessor::finish_cpu_read()
{
    impl_->finish_cpu_read();
}

cv::Mat nv12_dma_to_bgr_cpu(
    const vision_core::ImageInput& input)
{
    if (input.dma_fd < 0) {
        cv::Mat bgr;
        cv::cvtColor(input.image, bgr, cv::COLOR_YUV2BGR_NV12);
        return bgr;
    }
    dma_sync(
        input.dma_fd,
        DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
    cv::Mat bgr;
    try {
        cv::cvtColor(
            input.image, bgr, cv::COLOR_YUV2BGR_NV12);
    } catch (...) {
        dma_sync(
            input.dma_fd,
            DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
        throw;
    }
    dma_sync(
        input.dma_fd,
        DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
    return bgr;
}

}  // namespace vision_common
