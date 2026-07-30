/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_preprocess_geometry.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vision_operators {

FitResizeDimensions calculate_fit_resize_dimensions(
    float source_width,
    float source_height,
    int output_width,
    int output_height,
    PreprocessResizeRounding rounding)
{
    if (source_width <= 0.0F || source_height <= 0.0F ||
        output_width <= 0 || output_height <= 0) {
        throw std::runtime_error(
            "fit resize dimensions must be positive");
    }
    const float scale_x =
        static_cast<float>(output_width) / source_width;
    const float scale_y =
        static_cast<float>(output_height) / source_height;
    const bool width_limited = scale_x <= scale_y;
    const float scale = width_limited ? scale_x : scale_y;
    const auto convert = [rounding](float value) {
        return rounding == PreprocessResizeRounding::kTruncate
            ? static_cast<int>(value)
            : static_cast<int>(std::round(value));
    };
    const int width = width_limited
        ? output_width
        : convert(source_width * scale);
    const int height = width_limited
        ? convert(source_height * scale)
        : output_height;
    return {std::max(1, width), std::max(1, height)};
}

ImagePreprocessGeometry make_image_preprocess_geometry(
    const ImagePreprocessSpec& spec,
    int input_width,
    int input_height)
{
    if (input_width <= 0 || input_height <= 0) {
        throw std::runtime_error(
            "image preprocess input dimensions must be positive");
    }
    if (spec.output_width <= 0 || spec.output_height <= 0) {
        throw std::runtime_error(
            "image preprocess output dimensions must be positive");
    }

    ImagePreprocessGeometry geometry;
    geometry.src_width = static_cast<float>(input_width);
    geometry.src_height = static_cast<float>(input_height);
    geometry.dst_width = spec.output_width;
    geometry.dst_height = spec.output_height;

    if (spec.crop_mode == PreprocessCropMode::kCenterSquare) {
        const float side = static_cast<float>(
            std::min(input_width, input_height));
        geometry.src_x = (input_width - side) * 0.5F;
        geometry.src_y = (input_height - side) * 0.5F;
        geometry.src_width = side;
        geometry.src_height = side;
    } else if (
        spec.crop_mode ==
        PreprocessCropMode::kResizeShortSideCenterCrop) {
        if (spec.resize_width <= 0 || spec.resize_height <= 0) {
            throw std::runtime_error(
                "center-crop preprocessing requires resize dimensions");
        }
        const float virtual_scale_x =
            static_cast<float>(spec.resize_width) / input_width;
        const float virtual_scale_y =
            static_cast<float>(spec.resize_height) / input_height;
        geometry.src_width =
            static_cast<float>(spec.output_width) / virtual_scale_x;
        geometry.src_height =
            static_cast<float>(spec.output_height) / virtual_scale_y;
        geometry.src_x =
            (input_width - geometry.src_width) * 0.5F;
        geometry.src_y =
            (input_height - geometry.src_height) * 0.5F;
    }

    if (spec.resize_mode != PreprocessResizeMode::kStretch) {
        const FitResizeDimensions dimensions =
            calculate_fit_resize_dimensions(
                geometry.src_width,
                geometry.src_height,
                spec.output_width,
                spec.output_height,
                spec.resize_rounding);
        geometry.dst_width = dimensions.width;
        geometry.dst_height = dimensions.height;
        if (spec.resize_mode == PreprocessResizeMode::kLetterbox) {
            geometry.dst_x = static_cast<int>(std::round(
                (spec.output_width - geometry.dst_width) /
                    2.0F - 0.1F));
            geometry.dst_y = static_cast<int>(std::round(
                (spec.output_height - geometry.dst_height) /
                    2.0F - 0.1F));
        }
    }

    return geometry;
}

}  // namespace vision_operators
