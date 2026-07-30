/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMAGE_PREPROCESS_BACKEND_H
#define IMAGE_PREPROCESS_BACKEND_H

#include <string>
#include <string_view>

namespace vision_operators {

enum class PreprocessBackendPolicy {
    kCpu,
    kAuto,
    kOpenCl,
};

enum class PreprocessBackend {
    kCpu,
    kOpenCl,
};

PreprocessBackendPolicy parse_preprocess_backend_policy(
    std::string_view value);

class OpenClBackendState {
public:
    explicit OpenClBackendState(PreprocessBackendPolicy policy);

    PreprocessBackendPolicy policy() const noexcept;
    bool should_try_opencl() const noexcept;
    bool should_try_opencl_for_input(
        bool is_nv12,
        bool has_dma_fd) const noexcept;

    /**
     * Disable OpenCL after a backend failure.
     *
     * Returns true only for the first kAuto transition to disabled. Callers
     * use that transition to emit one warning. Strict kOpenCl is never
     * converted into fallback.
     */
    bool disable(std::string reason);

    const std::string& disable_reason() const noexcept;

private:
    PreprocessBackendPolicy policy_;
    bool disabled_{false};
    std::string disable_reason_;
};

}  // namespace vision_operators

#endif  // IMAGE_PREPROCESS_BACKEND_H
