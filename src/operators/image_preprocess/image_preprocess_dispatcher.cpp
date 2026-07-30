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

#include "image_preprocessor.h"

namespace vision_operators {

namespace {

bool same_preprocess_spec(
    const ImagePreprocessSpec& left,
    const ImagePreprocessSpec& right)
{
    return left.batch_size == right.batch_size &&
        left.output_width == right.output_width &&
        left.output_height == right.output_height &&
        left.crop_mode == right.crop_mode &&
        left.resize_mode == right.resize_mode &&
        left.resize_rounding == right.resize_rounding &&
        left.resize_width == right.resize_width &&
        left.resize_height == right.resize_height &&
        left.output_rgb == right.output_rgb &&
        left.interpolation == right.interpolation &&
        left.output_type == right.output_type &&
        left.mean == right.mean &&
        left.scale == right.scale &&
        left.padding == right.padding;
}

void validate_input(const vision_core::ImageInput& input)
{
    if (input.image.empty()) {
        throw std::invalid_argument(
            "image preprocess input is empty");
    }
    if (input.format == vision_core::ImagePixelFormat::kBgr8) {
        if (input.image.type() != CV_8UC3) {
            throw std::invalid_argument(
                "BGR8 input must have type CV_8UC3");
        }
    } else {
        const int input_height = input.image.rows * 2 / 3;
        if (input.image.type() != CV_8UC1 ||
            input.image.rows % 3 != 0 ||
            (input.image.cols & 1) != 0 ||
            (input_height & 1) != 0) {
            throw std::invalid_argument(
                "NV12 input must be CV_8UC1 H*3/2 x W "
                "with even H and W");
        }
    }
    if (input.image.step[0] == 0) {
        throw std::invalid_argument(
            "image preprocess input has an invalid row stride");
    }
    if (input.dma_fd >= 0) {
        struct stat info {};
        if (::fstat(input.dma_fd, &info) != 0) {
            throw std::invalid_argument(
                "invalid input dma-buf fd: " +
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
    if (spec.crop_mode ==
            PreprocessCropMode::kResizeShortSideCenterCrop &&
        (spec.resize_width <= 0 || spec.resize_height <= 0)) {
        throw std::invalid_argument(
            "center-crop preprocessing requires resize dimensions");
    }
}

}  // namespace

class ImagePreprocessDispatcher::Impl
    : public std::enable_shared_from_this<
        ImagePreprocessDispatcher::Impl> {
public:
    Impl(
        PreprocessBackendPolicy policy,
        ImagePreprocessorFactory opencl_factory)
        : state_(policy),
        opencl_factory_(std::move(opencl_factory))
    {
        if (!opencl_factory_) {
            throw std::invalid_argument(
                "OpenCL image preprocessor factory is empty");
        }
    }

    void configure(const std::string& backend)
    {
        PreprocessBackendPolicy policy =
            parse_preprocess_backend_policy(backend);
        if (!opencl_image_preprocessor_compiled()) {
            if (policy == PreprocessBackendPolicy::kOpenCl) {
                throw std::runtime_error(
                    "OpenCL image preprocessing was not compiled");
            }
            if (policy == PreprocessBackendPolicy::kAuto) {
                policy = PreprocessBackendPolicy::kCpu;
            }
        }
        state_ = OpenClBackendState(policy);
        reset_opencl();
    }

    void reset()
    {
        reset_opencl();
    }

    ImagePreprocessResult process(
        const vision_core::ImageInput& input,
        const ImagePreprocessSpec& spec,
        const CpuImagePreprocess& cpu_preprocess)
    {
        // Input errors are independent of backend capability and must never
        // disable OpenCL or trigger fallback.
        validate_input(input);

        const bool is_nv12 =
            input.format == vision_core::ImagePixelFormat::kNv12;
        const bool is_opencl_input =
            is_nv12 && input.dma_fd >= 0;
        if (state_.policy() ==
                PreprocessBackendPolicy::kOpenCl &&
            !is_opencl_input) {
            throw std::invalid_argument(
                "OpenCL image preprocessing requires "
                "NV12 DMA-BUF input");
        }
        if (!state_.should_try_opencl_for_input(
                is_nv12, input.dma_fd >= 0)) {
            return run_cpu_image_preprocess(
                input, cpu_preprocess);
        }
        validate_spec(spec);

        if (!opencl_preprocessor_ ||
            !has_opencl_spec_ ||
            !same_preprocess_spec(opencl_spec_, spec)) {
            try {
                opencl_preprocessor_ =
                    opencl_factory_(spec);
                opencl_spec_ = spec;
                has_opencl_spec_ = true;
            } catch (const std::exception& error) {
                reset_opencl();
                if (state_.policy() ==
                    PreprocessBackendPolicy::kOpenCl) {
                    throw;
                }
                disable_with_warning(error.what());
                return run_cpu_image_preprocess(
                    input, cpu_preprocess);
            }
        }

        try {
            cv::Mat tensor = opencl_preprocessor_->process(input);
            std::shared_ptr<ImagePreprocessor> retained =
                opencl_preprocessor_;
            std::shared_ptr<Impl> self =
                shared_from_this();
            return ImagePreprocessResult(
                std::move(tensor),
                PreprocessBackend::kOpenCl,
                [self, retained]() {
                    try {
                        retained->complete();
                    } catch (const std::exception& error) {
                        if (self->state_.policy() ==
                            PreprocessBackendPolicy::kAuto) {
                            self->disable_with_warning(
                                error.what());
                            self->reset_opencl();
                        }
                        throw;
                    }
                });
        } catch (const std::invalid_argument&) {
            // Request-specific input/layout errors do not indicate a
            // persistent OpenCL backend failure.
            throw;
        } catch (
            const ImagePreprocessBackendUnavailable& error) {
            if (state_.policy() ==
                PreprocessBackendPolicy::kOpenCl) {
                throw;
            }
            disable_with_warning(error.what());
            reset_opencl();
            return run_cpu_image_preprocess(
                input, cpu_preprocess);
        } catch (const std::exception& error) {
            if (state_.policy() == PreprocessBackendPolicy::kAuto) {
                disable_with_warning(error.what());
                reset_opencl();
            }
            // Once execution begins, the current request is never retried on
            // CPU because external-memory ownership may be indeterminate.
            throw;
        }
    }

private:
    void disable_with_warning(const std::string& reason)
    {
        if (!state_.disable(reason)) return;
        std::cerr
            << "[WARN] image_preprocess OpenCL disabled: "
            << reason
            << "; subsequent auto requests will use CPU\n";
    }

    void reset_opencl()
    {
        opencl_preprocessor_.reset();
        has_opencl_spec_ = false;
    }

    OpenClBackendState state_;
    ImagePreprocessorFactory opencl_factory_;
    std::shared_ptr<ImagePreprocessor> opencl_preprocessor_;
    ImagePreprocessSpec opencl_spec_;
    bool has_opencl_spec_{false};
};

ImagePreprocessDispatcher::ImagePreprocessDispatcher(
    PreprocessBackendPolicy policy)
    : ImagePreprocessDispatcher(
        policy,
        [](const ImagePreprocessSpec& spec) {
            return create_opencl_image_preprocessor(spec);
        })
{
}

ImagePreprocessDispatcher::ImagePreprocessDispatcher(
    PreprocessBackendPolicy policy,
    ImagePreprocessorFactory opencl_factory)
    : impl_(std::make_shared<Impl>(
        policy, std::move(opencl_factory)))
{
}

ImagePreprocessDispatcher::~ImagePreprocessDispatcher() = default;

void ImagePreprocessDispatcher::configure(
    const std::string& backend)
{
    impl_->configure(backend);
}

void ImagePreprocessDispatcher::reset()
{
    impl_->reset();
}

ImagePreprocessResult ImagePreprocessDispatcher::process(
    const vision_core::ImageInput& input,
    const ImagePreprocessSpec& spec,
    const CpuImagePreprocess& cpu_preprocess)
{
    return impl_->process(input, spec, cpu_preprocess);
}

}  // namespace vision_operators
