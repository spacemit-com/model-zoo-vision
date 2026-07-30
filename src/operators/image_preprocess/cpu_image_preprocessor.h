/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CPU_IMAGE_PREPROCESSOR_H
#define CPU_IMAGE_PREPROCESSOR_H

#include <functional>

#include <opencv2/core.hpp>

#include "image_preprocess_result.h"
#include "core/cpp/vision_infer_types.h"

namespace vision_operators {

using CpuImagePreprocess =
    std::function<cv::Mat(const cv::Mat&)>;

cv::Mat image_input_to_bgr_cpu(
    const vision_core::ImageInput& input);

ImagePreprocessResult run_cpu_image_preprocess(
    const vision_core::ImageInput& input,
    const CpuImagePreprocess& cpu_preprocess);

}  // namespace vision_operators

#endif  // CPU_IMAGE_PREPROCESSOR_H
