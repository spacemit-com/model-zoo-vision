/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_preprocess_backend.h"

#include <stdexcept>
#include <utility>

namespace vision_operators {

PreprocessBackendPolicy parse_preprocess_backend_policy(
    std::string_view value)
{
    if (value == "cpu") {
        return PreprocessBackendPolicy::kCpu;
    }
    if (value == "auto") {
        return PreprocessBackendPolicy::kAuto;
    }
    if (value == "opencl") {
        return PreprocessBackendPolicy::kOpenCl;
    }
    if (value == "v2d") return PreprocessBackendPolicy::kV2d;
    throw std::invalid_argument(
        "preprocess.backend must be cpu, auto, opencl, or v2d");
}

PreprocessFallback parse_preprocess_fallback(std::string_view value)
{
    if (value == "cpu") return PreprocessFallback::kCpu;
    if (value == "error") return PreprocessFallback::kError;
    throw std::invalid_argument("preprocess.fallback must be cpu or error");
}

const char* preprocess_backend_name(PreprocessBackend backend) noexcept
{
    switch (backend) {
    case PreprocessBackend::kCpu: return "cpu";
    case PreprocessBackend::kOpenCl: return "opencl";
    case PreprocessBackend::kV2d: return "v2d";
    }
    return "unknown";
}

PreprocessOpenClSampling parse_preprocess_opencl_sampling(
    std::string_view value)
{
    if (value == "opencv_compatible") {
        return PreprocessOpenClSampling::kOpenCvCompatible;
    }
    if (value == "fast") {
        return PreprocessOpenClSampling::kFast;
    }
    throw std::invalid_argument(
        "preprocess.opencl_sampling must be "
        "opencv_compatible or fast");
}

OpenClBackendState::OpenClBackendState(
    PreprocessBackendPolicy policy)
    : policy_(policy)
{
}

PreprocessBackendPolicy OpenClBackendState::policy() const noexcept
{
    return policy_;
}

bool OpenClBackendState::should_try_opencl() const noexcept
{
    return (policy_ == PreprocessBackendPolicy::kAuto ||
            policy_ == PreprocessBackendPolicy::kOpenCl) &&
        !disabled_;
}

bool OpenClBackendState::should_try_opencl_for_input(
    bool is_nv12,
    bool has_dma_fd) const noexcept
{
    return should_try_opencl() && is_nv12 && has_dma_fd;
}

bool OpenClBackendState::disable(std::string reason)
{
    if (policy_ != PreprocessBackendPolicy::kAuto || disabled_) {
        return false;
    }
    disabled_ = true;
    disable_reason_ = std::move(reason);
    return true;
}

const std::string& OpenClBackendState::disable_reason() const noexcept
{
    return disable_reason_;
}

}  // namespace vision_operators
