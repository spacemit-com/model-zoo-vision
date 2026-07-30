/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "backends/opencl/operators/opencl_image_preprocessor.h"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "backends/opencl/memory/dma_buffer.h"
#include "backends/opencl/memory/dmabuf_import_cache.h"
#include "backends/opencl/memory/external_memory_guard.h"
#include "backends/opencl/memory/opencl_buffer.h"
#include "backends/opencl/runtime/opencl_context.h"
#include "backends/opencl/runtime/opencl_error.h"
#include "backends/opencl/runtime/opencl_handles.h"
#include "generated/opencl/image_preprocess_kernel.inc"
#include "operators/image_preprocess/image_preprocess_geometry.h"

namespace vision_opencl {
namespace {

const char* kKernelSource =
    vision_opencl_embedded::kImagePreprocessKernelSource;

using GeometryPlan =
    vision_operators::ImagePreprocessGeometry;
using vision_operators::ImagePreprocessSpec;
using vision_operators::PreprocessInterpolation;
using vision_operators::PreprocessOutputType;

cl_float4 make_float4(const std::array<float, 3>& values)
{
    cl_float4 result{};
    result.s[0] = values[0];
    result.s[1] = values[1];
    result.s[2] = values[2];
    return result;
}

GeometryPlan make_geometry(
    const ImagePreprocessSpec& spec,
    int input_width,
    int input_height)
{
    return vision_operators::make_image_preprocess_geometry(
        spec, input_width, input_height);
}

}  // namespace

class OpenClImagePreprocessor::Impl {
public:
    struct OutputSlot {
        std::unique_ptr<DmaBuffer> buffer;
        OpenClBuffer memory;
    };

    Impl(
        const ImagePreprocessSpec& spec,
        int output_ring_depth)
        : spec_(spec),
        runtime_(OpenClContext::shared()),
        input_cache_(runtime_)
    {
        validate_spec(output_ring_depth);
        queue_.reset(runtime_->create_queue());
        initialize_kernels();
        initialize_output_ring(output_ring_depth);
    }

    ~Impl()
    {
        try {
            finish_cpu_read();
        } catch (...) {
        }
    }

    cv::Mat process(const vision_core::ImageInput& input)
    {
        validate_input(input);
        finish_cpu_read();

        output_index_ =
            (output_index_ + 1U) % output_ring_.size();
        OutputSlot& output = output_ring_[output_index_];
        ImportedNv12DmaBuffer* imported = nullptr;
        GeometryPlan geometry;
        try {
            imported = &input_cache_.get(input);
            geometry = make_geometry(
                spec_,
                imported->identity.width,
                imported->identity.height);
        } catch (const std::invalid_argument&) {
            throw;
        } catch (const std::exception& error) {
            throw vision_operators::
                ImagePreprocessBackendUnavailable(
                    error.what());
        }

        ExternalMemoryGuard external_memory(
            runtime_,
            queue_.get(),
            {imported->buffer.get(), output.memory.get()});
        enqueue(*imported, output.memory.get(), geometry);
        external_memory.release();
        check_cl(clFinish(queue_.get()), "clFinish");

        output.buffer->start_cpu_read();
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
        if (active_output_index_ >= output_ring_.size()) {
            return;
        }
        output_ring_[active_output_index_]
            .buffer->end_cpu_read();
        active_output_index_ = output_ring_.size();
    }

private:
    enum KernelIndex {
        kNv12ImagesFloat32,
        kNv12ImagesFloat16,
        kKernelCount,
    };

    void validate_spec(int output_ring_depth) const
    {
        if (spec_.batch_size != 1) {
            throw std::runtime_error(
                "OpenCL image preprocessing supports batch size 1");
        }
        if (spec_.output_width <= 0 ||
            spec_.output_height <= 0) {
            throw std::runtime_error(
                "OpenCL preprocess output dimensions "
                "must be positive");
        }
        if (output_ring_depth < 2 ||
            output_ring_depth > 16) {
            throw std::runtime_error(
                "OpenCL preprocess output ring depth "
                "must be in [2, 16]");
        }
    }

    void validate_input(
        const vision_core::ImageInput& input) const
    {
        if (input.image.empty()) {
            throw std::invalid_argument(
                "OpenCL preprocess input is empty");
        }
        const int input_height =
            input.image.rows * 2 / 3;
        if (input.format !=
                vision_core::ImagePixelFormat::kNv12 ||
            input.dma_fd < 0 ||
            input.image.type() != CV_8UC1 ||
            input.image.rows % 3 != 0 ||
            (input.image.cols & 1) != 0 ||
            (input_height & 1) != 0 ||
            input.image.step[0] == 0) {
            throw std::invalid_argument(
                "OpenCL image preprocessing requires "
                "NV12 DMA-BUF input with even dimensions");
        }
    }

    void initialize_kernels()
    {
        cl_program program =
            runtime_->program_cache().get_or_build(
                kKernelSource,
                "-cl-fast-relaxed-math");
        const char* names[] = {
            "preprocess_nv12_images_f32",
            "preprocess_nv12_images_f16"};
        for (int i = 0; i < kKernelCount; ++i) {
            cl_int error = CL_SUCCESS;
            kernels_[i].reset(
                clCreateKernel(program, names[i], &error));
            if (error == CL_SUCCESS) continue;
            if (i == kNv12ImagesFloat16 &&
                spec_.output_type ==
                    PreprocessOutputType::kFloat32) {
                kernels_[i].reset();
                continue;
            }
            check_cl(error, "clCreateKernel");
        }
    }

    void initialize_output_ring(int depth)
    {
        const size_t element_size =
            spec_.output_type ==
                PreprocessOutputType::kFloat32
            ? sizeof(float)
            : sizeof(uint16_t);
        const size_t output_size =
            static_cast<size_t>(spec_.output_width) *
            spec_.output_height * 3U * element_size;
        output_ring_.reserve(static_cast<size_t>(depth));
        for (int i = 0; i < depth; ++i) {
            OutputSlot slot;
            slot.buffer =
                std::make_unique<DmaBuffer>(output_size);
            slot.memory = import_dma_buffer(
                runtime_,
                slot.buffer->fd(),
                slot.buffer->size());
            output_ring_.push_back(std::move(slot));
        }
        output_index_ = output_ring_.size() - 1U;
        active_output_index_ = output_ring_.size();
    }

    void enqueue(
        const ImportedNv12DmaBuffer& input,
        cl_mem output,
        const GeometryPlan& geometry)
    {
        const bool fp16 =
            spec_.output_type ==
            PreprocessOutputType::kFloat16;
        cl_kernel kernel = kernels_[
            fp16
                ? kNv12ImagesFloat16
                : kNv12ImagesFloat32].get();
        if (!kernel) {
            throw std::runtime_error(
                "requested OpenCL output type "
                "is unsupported");
        }

        const int input_width = input.identity.width;
        const int input_height = input.identity.height;
        cl_mem y_image = input.y_image.get();
        cl_mem uv_image = input.uv_image.get();
        cl_int error = CL_SUCCESS;
        int argument = 0;
        error |= clSetKernelArg(
            kernel, argument++, sizeof(cl_mem), &y_image);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(cl_mem), &uv_image);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(cl_mem), &output);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &input_width);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &input_height);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int),
            &spec_.output_width);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int),
            &spec_.output_height);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(float),
            &geometry.src_x);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(float),
            &geometry.src_y);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(float),
            &geometry.src_width);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(float),
            &geometry.src_height);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int),
            &geometry.dst_x);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int),
            &geometry.dst_y);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int),
            &geometry.dst_width);
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int),
            &geometry.dst_height);
        const int output_rgb = spec_.output_rgb ? 1 : 0;
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int), &output_rgb);
        const int interpolation =
            spec_.interpolation ==
                PreprocessInterpolation::kNearest
            ? 1
            : 0;
        error |= clSetKernelArg(
            kernel, argument++, sizeof(int),
            &interpolation);
        const cl_float4 mean = make_float4(spec_.mean);
        const cl_float4 scale = make_float4(spec_.scale);
        const cl_float4 padding =
            make_float4(spec_.padding);
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
                queue_.get(),
                kernel,
                2,
                nullptr,
                global,
                nullptr,
                0,
                nullptr,
                nullptr),
            "clEnqueueNDRangeKernel");
    }

    ImagePreprocessSpec spec_;
    std::shared_ptr<OpenClContext> runtime_;
    DmaBufImportCache input_cache_;
    std::vector<OutputSlot> output_ring_;
    size_t output_index_{0};
    size_t active_output_index_{0};
    OpenClCommandQueue queue_;
    OpenClKernel kernels_[kKernelCount];
};

OpenClImagePreprocessor::OpenClImagePreprocessor(
    const ImagePreprocessSpec& spec,
    int output_ring_depth)
    : impl_(std::make_unique<Impl>(
        spec, output_ring_depth))
{
}

OpenClImagePreprocessor::~OpenClImagePreprocessor() =
    default;

cv::Mat OpenClImagePreprocessor::process(
    const vision_core::ImageInput& input)
{
    return impl_->process(input);
}

void OpenClImagePreprocessor::finish_cpu_read()
{
    impl_->finish_cpu_read();
}

void OpenClImagePreprocessor::complete()
{
    finish_cpu_read();
}

}  // namespace vision_opencl

namespace vision_operators {

std::shared_ptr<ImagePreprocessor>
create_opencl_image_preprocessor(
    const ImagePreprocessSpec& spec,
    int output_ring_depth)
{
    return std::make_shared<
        vision_opencl::OpenClImagePreprocessor>(
            spec, output_ring_depth);
}

bool opencl_image_preprocessor_compiled() noexcept
{
    return true;
}

}  // namespace vision_operators
