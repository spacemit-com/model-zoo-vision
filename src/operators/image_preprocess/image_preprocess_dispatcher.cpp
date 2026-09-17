/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_preprocess_dispatcher.h"

#include <cerrno>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <utility>
#include <vector>

#include "image_preprocessor.h"

namespace vision_operators
{

namespace
{

bool same_preprocess_spec(const ImagePreprocessSpec& left,
                            const ImagePreprocessSpec& right)
{
    return left.batch_size == right.batch_size &&
            left.output_width == right.output_width &&
            left.output_height == right.output_height &&
            left.crop_mode == right.crop_mode && left.resize_mode == right.resize_mode &&
            left.resize_rounding == right.resize_rounding &&
            left.resize_width == right.resize_width &&
            left.resize_height == right.resize_height &&
            left.output_rgb == right.output_rgb &&
            left.interpolation == right.interpolation &&
            left.opencl_sampling == right.opencl_sampling &&
            left.output_type == right.output_type && left.mean == right.mean &&
            left.scale == right.scale && left.padding == right.padding;
}

void validate_input(const vision_core::ImageInput& input)
{
    if (input.image.empty()) {
        throw std::invalid_argument("image preprocess input is empty");
    }
    if (input.format == vision_core::ImagePixelFormat::kBgr8) {
        if (input.image.type() != CV_8UC3) {
            throw std::invalid_argument("BGR8 input must have type CV_8UC3");
        }
    } else {
        const int input_height = input.image.rows * 2 / 3;
        if (input.image.type() != CV_8UC1 || input.image.rows % 3 != 0 ||
            (input.image.cols & 1) != 0 || (input_height & 1) != 0) {
            throw std::invalid_argument(
                "NV12 input must be CV_8UC1 H*3/2 x W "
                "with even H and W");
        }
    }
    if (input.image.step[0] == 0) {
        throw std::invalid_argument("image preprocess input has an invalid row stride");
    }
    if (input.dma_fd >= 0) {
        struct stat info{};
        if (::fstat(input.dma_fd, &info) != 0) {
            throw std::invalid_argument("invalid input dma-buf fd: " +
                                        std::string(std::strerror(errno)));
        }
    }
}

void validate_spec(const ImagePreprocessSpec& spec)
{
    if (spec.output_width <= 0 || spec.output_height <= 0) {
        throw std::invalid_argument(
            "image preprocess output dimensions must be positive");
    }
    if (spec.crop_mode == PreprocessCropMode::kResizeShortSideCenterCrop &&
        (spec.resize_width <= 0 || spec.resize_height <= 0)) {
        throw std::invalid_argument(
            "center-crop preprocessing requires resize dimensions");
    }
}

}  // namespace

class ImagePreprocessDispatcher::Impl
    : public std::enable_shared_from_this<ImagePreprocessDispatcher::Impl>
{
    struct Backend {
        PreprocessBackend kind;
        ImagePreprocessorFactory factory;
        bool compiled;
        bool disabled = false;
        bool has_spec = false;
        bool warned = false;
        ImagePreprocessSpec spec;
        std::shared_ptr<ImagePreprocessor> processor;
    };

public:
    Impl(PreprocessBackendPolicy policy, ImagePreprocessorFactory opencl_factory,
            bool with_v2d)
        : policy_(policy)
    {
        if (!opencl_factory) throw std::invalid_argument("empty OpenCL factory");
        // The injectable constructor deliberately isolates OpenCL for existing
        // callers/tests. Production registers each backend here, not in models.
        if (with_v2d)
            backends_.push_back({PreprocessBackend::kV2d, create_v2d_image_preprocessor,
                                    v2d_image_preprocessor_compiled()});
        backends_.push_back({PreprocessBackend::kOpenCl, std::move(opencl_factory),
                                opencl_image_preprocessor_compiled()});
    }

    void configure(const std::string& name)
    {
        const auto policy = parse_preprocess_backend_policy(name);
        if (policy != PreprocessBackendPolicy::kCpu &&
            policy != PreprocessBackendPolicy::kAuto &&
            fallback_ == PreprocessFallback::kError) {
            bool available = false;
            for (const auto& b : backends_)
                if (selected(policy, b.kind)) available = b.compiled;
            if (!available)
                throw std::runtime_error(name + " preprocessing was not compiled");
        }
        policy_ = policy;
        for (auto& b : backends_) {
            b.disabled = false;
            b.warned = false;
        }
        reset();
    }

    void configure_fallback(const std::string& value)
    {
        fallback_ = parse_preprocess_fallback(value);
    }

    void reset()
    {
        for (auto& b : backends_) {
            b.processor.reset();
            b.has_spec = false;
        }
    }

    ImagePreprocessResult process(const vision_core::ImageInput& input,
                                    const ImagePreprocessSpec& spec,
                                    const CpuImagePreprocess& cpu)
    {
        validate_input(input);
        if (policy_ == PreprocessBackendPolicy::kCpu)
            return run_cpu_image_preprocess(input, cpu);
        const bool automatic = policy_ == PreprocessBackendPolicy::kAuto;
        const bool nv12_dma =
            input.format == vision_core::ImagePixelFormat::kNv12 && input.dma_fd >= 0;
        // Auto never uploads host BGR, or converts BGR to NV12 merely to use V2D.
        if (automatic && !nv12_dma) return run_cpu_image_preprocess(input, cpu);
        validate_spec(spec);
        const bool can_fallback = automatic || fallback_ == PreprocessFallback::kCpu;
        for (size_t i = 0; i < backends_.size(); ++i) {
            auto& b = backends_[i];
            if (!automatic && !selected(policy_, b.kind)) continue;
            if (!b.compiled || b.disabled) {
                if (!can_fallback)
                    throw ImagePreprocessBackendUnavailable(
                        std::string(preprocess_backend_name(b.kind)) + " unavailable");
                if (!automatic) warn(b, "backend unavailable");
                continue;
            }
            const bool supported_input =
                nv12_dma || (b.kind == PreprocessBackend::kOpenCl &&
                                input.format == vision_core::ImagePixelFormat::kBgr8);
            if (!supported_input) {
                if (!can_fallback)
                    throw std::invalid_argument(
                        std::string(preprocess_backend_name(b.kind)) +
                        " does not support this input (V2D requires NV12 DMA)");
                warn(b, "input unsupported; using CPU");
                continue;
            }
            // Construction does not own the input buffer: safe to fall back.
            try {
                if (!b.processor || !b.has_spec ||
                    !same_preprocess_spec(b.spec, spec)) {
                    b.processor = b.factory(spec);
                    if (!b.processor)
                        throw std::runtime_error("empty backend instance");
                    b.spec = spec;
                    b.has_spec = true;
                }
            } catch (const std::invalid_argument&) {
                throw;
            } catch (const ImagePreprocessUnsupported& e) {
                b.processor.reset();
                b.has_spec = false;
                if (!can_fallback) throw;
                warn(b, e.what());
                continue;
            } catch (const std::exception& e) {
                if (!can_fallback) throw;
                disable(b, e.what());
                continue;
            }
            try {
                cv::Mat tensor = b.processor->process(input);
                auto retained = b.processor;
                auto self = shared_from_this();
                return ImagePreprocessResult(
                    std::move(tensor), b.kind, [self, retained, i, can_fallback]() {
                        try {
                            retained->complete();
                        } catch (const std::exception& e) {
                            if (can_fallback)
                                self->disable(self->backends_[i], e.what());
                            throw;
                        }
                    });
            } catch (const std::invalid_argument&) {
                throw;  // Invalid caller data is never a fallback condition.
            } catch (const ImagePreprocessUnsupported& e) {
                if (!can_fallback) throw;
                warn(b, e.what());
            } catch (const ImagePreprocessBackendUnavailable& e) {
                // Backend promises no input acquisition/work was started.
                if (!can_fallback) throw;
                disable(b, e.what());
            } catch (const std::exception& e) {
                if (can_fallback) disable(b, e.what());
                // Once work starts, ownership may be indeterminate. Never
                // retry this request on another backend or on CPU.
                throw;
            }
        }
        return run_cpu_image_preprocess(input, cpu);
    }

private:
    static bool selected(PreprocessBackendPolicy p, PreprocessBackend k)
    {
        return (p == PreprocessBackendPolicy::kOpenCl &&
                k == PreprocessBackend::kOpenCl) ||
                (p == PreprocessBackendPolicy::kV2d && k == PreprocessBackend::kV2d);
    }
    static void warn(Backend& b, const std::string& reason)
    {
        if (b.warned) return;
        b.warned = true;
        std::cerr << "[WARN] image_preprocess " << preprocess_backend_name(b.kind)
                    << ": " << reason << "; fallback enabled\n";
    }
    static void disable(Backend& b, const std::string& reason)
    {
        b.disabled = true;
        b.processor.reset();
        b.has_spec = false;
        warn(b, reason + (b.kind == PreprocessBackend::kOpenCl
                                ? "; subsequent auto requests will use CPU"
                                : "; subsequent requests skip this backend"));
    }
    PreprocessBackendPolicy policy_;
    PreprocessFallback fallback_ = PreprocessFallback::kError;
    std::vector<Backend> backends_;
};

ImagePreprocessDispatcher::ImagePreprocessDispatcher(PreprocessBackendPolicy policy)
    : impl_(std::make_shared<Impl>(
            policy,
            [](const ImagePreprocessSpec& spec) {
                return create_opencl_image_preprocessor(spec);
            },
            true))
{
}

ImagePreprocessDispatcher::ImagePreprocessDispatcher(
    PreprocessBackendPolicy policy, ImagePreprocessorFactory opencl_factory)
    : impl_(std::make_shared<Impl>(policy, std::move(opencl_factory), false))
{
}

ImagePreprocessDispatcher::~ImagePreprocessDispatcher() = default;

void ImagePreprocessDispatcher::configure(const std::string& backend)
{
    impl_->configure(backend);
}

void ImagePreprocessDispatcher::configure_fallback(const std::string& fallback)
{
    impl_->configure_fallback(fallback);
}

void ImagePreprocessDispatcher::reset() { impl_->reset(); }

ImagePreprocessResult ImagePreprocessDispatcher::process(
    const vision_core::ImageInput& input, const ImagePreprocessSpec& spec,
    const CpuImagePreprocess& cpu_preprocess)
{
    return impl_->process(input, spec, cpu_preprocess);
}

}  // namespace vision_operators
