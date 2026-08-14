/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CPU_IMAGE_PREPROCESSOR_H
#define CPU_IMAGE_PREPROCESSOR_H

#include <array>
#include <functional>

#include <opencv2/core.hpp>

#include "image_preprocess_result.h"
#include "image_preprocess_spec.h"
#include "core/cpp/vision_infer_types.h"

namespace vision_operators {

using CpuImagePreprocess =
    std::function<cv::Mat(const cv::Mat&)>;

struct CpuChannelTransform {
    std::array<float, 3> input_scale{1.0F, 1.0F, 1.0F};
    // Optional explicit divisors preserve pipelines written as value / c
    // instead of replacing division with multiplication by a reciprocal.
    std::array<float, 3> input_divisor{1.0F, 1.0F, 1.0F};
    std::array<float, 3> mean{0.0F, 0.0F, 0.0F};
    std::array<float, 3> output_scale{1.0F, 1.0F, 1.0F};
    std::array<float, 3> output_divisor{1.0F, 1.0F, 1.0F};
};

struct CpuGrayscaleTransform {
    // Source-channel weights are ordered B, G, R to match the input image.
    std::array<float, 3> bgr_weights{0.114F, 0.587F, 0.299F};
    float input_scale{1.0F};
    float mean{0.0F};
    float output_scale{1.0F};
};

cv::Mat image_input_to_bgr_cpu(
    const vision_core::ImageInput& input);

// Resize/crop a BGR8 image and fuse channel ordering, normalization, padding,
// and HWC-to-NCHW packing into the final tensor write. The returned tensor is
// an independent float32 [1,3,H,W] cv::Mat.
cv::Mat preprocess_bgr_to_nchw(
    const cv::Mat& bgr,
    const ImagePreprocessSpec& spec);

// Variant for models whose reference implementation first scales uint8 input
// (for example by 1/255), then subtracts mean and applies a second scale. This
// preserves the original floating-point operation order while retaining the
// fused tensor write.
cv::Mat preprocess_bgr_to_nchw(
    const cv::Mat& bgr,
    const ImagePreprocessSpec& spec,
    const CpuChannelTransform& transform);

// Resize/crop a BGR8 image and directly write a normalized grayscale
// float32 [1,1,H,W] tensor. Channel conversion and tensor packing are fused
// into the final write, avoiding intermediate float BGR and split planes.
cv::Mat preprocess_bgr_to_gray_nchw(
    const cv::Mat& bgr,
    const ImagePreprocessSpec& spec,
    const CpuGrayscaleTransform& transform = {});

ImagePreprocessResult run_cpu_image_preprocess(
    const vision_core::ImageInput& input,
    const CpuImagePreprocess& cpu_preprocess);

}  // namespace vision_operators

#endif  // CPU_IMAGE_PREPROCESSOR_H
