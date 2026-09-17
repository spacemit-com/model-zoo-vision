/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef V2D_IMAGE_OPS_H
#define V2D_IMAGE_OPS_H
#include "dma_buffer.h"
#include <opencv2/core.hpp>
#include <stdexcept>

namespace vision_spacemit
{
// Raised only before hardware submission; retrying on another backend is safe.
class V2dUnavailable : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};
// Current input contract: contiguous Y/UV in one DMA-BUF, equal strides,
// UV at stride*height; limited-range BT.601, matching the CPU NV12 path.
// No model geometry, normalization, or tensor ownership in this layer.
void resize_nv12_to_rgb(const cv::Mat& nv12, int input_fd, DmaBuffer& output, int width,
                        int height, int output_stride);
}  // namespace vision_spacemit

#endif  // V2D_IMAGE_OPS_H
