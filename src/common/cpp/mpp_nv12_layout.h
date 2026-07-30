/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MPP_NV12_LAYOUT_H
#define MPP_NV12_LAYOUT_H

#include <cstddef>
#include <cstdint>

namespace vision_mpp {

struct MppNv12Layout {
    int width = 0;
    int height = 0;
    int y_stride = 0;
    int uv_stride = 0;  // 0 means unspecified; inherit y_stride.
    size_t y_plane_size = 0;
    size_t uv_plane_size = 0;
    size_t total_size = 0;
    uintptr_t y_address = 0;
    uintptr_t uv_address = 0;
    int y_dma_fd = -1;
    int uv_dma_fd = -1;
};

bool is_importable_nv12_dma_layout(
    const MppNv12Layout& layout) noexcept;

}  // namespace vision_mpp

#endif  // MPP_NV12_LAYOUT_H
