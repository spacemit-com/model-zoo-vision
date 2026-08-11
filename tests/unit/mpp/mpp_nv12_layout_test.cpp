/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <iostream>

#include "mpp_nv12_layout.h"

namespace {

int failures = 0;

void check(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main()
{
    using vision_mpp::MppNv12Layout;
    using vision_mpp::is_importable_nv12_dma_layout;

    MppNv12Layout layout;
    layout.width = 1920;
    layout.height = 1080;
    layout.y_stride = 1920;
    layout.uv_stride = 1920;
    layout.y_plane_size = 1920U * 1080U;
    layout.uv_plane_size = 1920U * 1080U / 2U;
    layout.total_size =
        layout.y_plane_size + layout.uv_plane_size;
    layout.y_address = 0x10000000U;
    layout.uv_address =
        layout.y_address + 1920U * 1080U;
    layout.y_dma_fd = 7;
    layout.uv_dma_fd = 7;
    check(
        is_importable_nv12_dma_layout(layout),
        "contiguous NV12 DMA layout is importable");

    MppNv12Layout unspecified_uv_stride = layout;
    unspecified_uv_stride.uv_stride = 0;
    check(
        is_importable_nv12_dma_layout(unspecified_uv_stride),
        "unspecified UV stride inherits the Y stride");

    MppNv12Layout split_planes = layout;
    split_planes.uv_address += 4096U;
    check(
        !is_importable_nv12_dma_layout(split_planes),
        "non-contiguous UV plane is rejected");

    MppNv12Layout different_stride = layout;
    different_stride.uv_stride = 2048;
    check(
        !is_importable_nv12_dma_layout(different_stride),
        "different Y and UV strides are rejected");

    MppNv12Layout different_fds = layout;
    different_fds.uv_dma_fd = 8;
    check(
        !is_importable_nv12_dma_layout(different_fds),
        "different Y and UV DMA fds are rejected");

    MppNv12Layout short_y_plane = layout;
    --short_y_plane.y_plane_size;
    check(
        !is_importable_nv12_dma_layout(short_y_plane),
        "short Y plane is rejected");

    MppNv12Layout short_uv_plane = layout;
    --short_uv_plane.uv_plane_size;
    check(
        !is_importable_nv12_dma_layout(short_uv_plane),
        "short UV plane is rejected");

    MppNv12Layout short_buffer = layout;
    --short_buffer.total_size;
    check(
        !is_importable_nv12_dma_layout(short_buffer),
        "short DMA allocation is rejected");

    MppNv12Layout odd_width = layout;
    odd_width.width = 1919;
    check(
        !is_importable_nv12_dma_layout(odd_width),
        "odd NV12 width is rejected");

    MppNv12Layout invalid_fd = layout;
    invalid_fd.y_dma_fd = -1;
    check(
        !is_importable_nv12_dma_layout(invalid_fd),
        "invalid DMA fd is rejected");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: MPP NV12 DMA layout\n";
    return 0;
}
