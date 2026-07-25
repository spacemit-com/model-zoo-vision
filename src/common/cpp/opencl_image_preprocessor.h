/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENCL_IMAGE_PREPROCESSOR_H
#define OPENCL_IMAGE_PREPROCESSOR_H

#include <array>
#include <memory>

#include <opencv2/core.hpp>

#include "vision_infer_types.h"

namespace vision_common {

enum class PreprocessCropMode {
    kNone,
    kCenterSquare,
    kResizeShortSideCenterCrop,
};

enum class PreprocessResizeMode {
    kStretch,
    kLetterbox,
    kFitTopLeft,
};

enum class PreprocessOutputType {
    kFloat32,
    kFloat16,
};

enum class PreprocessInterpolation {
    kBilinear,
    kNearest,
};

struct OpenClPreprocessSpec {
    int output_width = 0;
    int output_height = 0;
    PreprocessCropMode crop_mode = PreprocessCropMode::kNone;
    PreprocessResizeMode resize_mode = PreprocessResizeMode::kStretch;
    int resize_width = 0;
    int resize_height = 0;
    bool output_rgb = true;
    PreprocessInterpolation interpolation =
        PreprocessInterpolation::kBilinear;
    PreprocessOutputType output_type = PreprocessOutputType::kFloat32;
    std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
    std::array<float, 3> scale{1.0F, 1.0F, 1.0F};
    std::array<float, 3> padding{0.0F, 0.0F, 0.0F};
};

class OpenClImagePreprocessor {
public:
    explicit OpenClImagePreprocessor(
        const OpenClPreprocessSpec& spec,
        int output_ring_depth = 3);
    ~OpenClImagePreprocessor();

    OpenClImagePreprocessor(const OpenClImagePreprocessor&) = delete;
    OpenClImagePreprocessor& operator=(const OpenClImagePreprocessor&) = delete;

    cv::Mat process(const vision_core::ImageInput& input);
    void finish_cpu_read();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

cv::Mat nv12_dma_to_bgr_cpu(const vision_core::ImageInput& input);

}  // namespace vision_common

#endif  // OPENCL_IMAGE_PREPROCESSOR_H
