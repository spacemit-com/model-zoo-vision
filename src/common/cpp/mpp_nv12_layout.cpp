/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mpp_nv12_layout.h"

#include <limits>

namespace vision_mpp {

bool is_importable_nv12_dma_layout(
    const MppNv12Layout& layout) noexcept
{
    const int uv_stride =
        layout.uv_stride == 0
        ? layout.y_stride
        : layout.uv_stride;
    if (layout.width <= 0 || layout.height <= 0 ||
        (layout.width & 1) != 0 || (layout.height & 1) != 0 ||
        layout.y_stride < layout.width ||
        uv_stride != layout.y_stride ||
        layout.y_address == 0 || layout.uv_address == 0 ||
        layout.y_dma_fd < 0 ||
        layout.uv_dma_fd != layout.y_dma_fd) {
        return false;
    }

    const size_t stride = static_cast<size_t>(layout.y_stride);
    const size_t height = static_cast<size_t>(layout.height);
    if (stride > std::numeric_limits<size_t>::max() / height) {
        return false;
    }
    const size_t y_size = stride * height;
    const size_t uv_height = height / 2U;
    if (stride > std::numeric_limits<size_t>::max() / uv_height) {
        return false;
    }
    const size_t uv_size = stride * uv_height;
    if (y_size > std::numeric_limits<size_t>::max() - uv_size ||
        layout.y_plane_size < y_size ||
        layout.uv_plane_size < uv_size ||
        layout.total_size < y_size + uv_size ||
        layout.y_address >
            std::numeric_limits<uintptr_t>::max() - y_size) {
        return false;
    }
    return layout.uv_address == layout.y_address + y_size;
}

}  // namespace vision_mpp
