/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMAGE_PREPROCESS_SPEC_H
#define IMAGE_PREPROCESS_SPEC_H

#include <array>

#include "image_preprocess_backend.h"

namespace vision_operators {

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

enum class PreprocessResizeRounding {
    kRound,
    kTruncate,
};

enum class PreprocessOutputType {
    kFloat32,
    kFloat16,
};

enum class PreprocessInterpolation {
    kBilinear,
    kNearest,
};

struct ImagePreprocessSpec {
    int batch_size = 1;
    int output_width = 0;
    int output_height = 0;
    PreprocessCropMode crop_mode = PreprocessCropMode::kNone;
    PreprocessResizeMode resize_mode = PreprocessResizeMode::kStretch;
    PreprocessResizeRounding resize_rounding =
        PreprocessResizeRounding::kRound;
    int resize_width = 0;
    int resize_height = 0;
    bool output_rgb = true;
    PreprocessInterpolation interpolation =
        PreprocessInterpolation::kBilinear;
    PreprocessOpenClSampling opencl_sampling =
        PreprocessOpenClSampling::kOpenCvCompatible;
    PreprocessOutputType output_type = PreprocessOutputType::kFloat32;
    std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
    std::array<float, 3> scale{1.0F, 1.0F, 1.0F};
    std::array<float, 3> padding{0.0F, 0.0F, 0.0F};
};

}  // namespace vision_operators

#endif  // IMAGE_PREPROCESS_SPEC_H
