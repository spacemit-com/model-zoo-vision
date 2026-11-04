/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMAGE_PREPROCESS_GEOMETRY_H
#define IMAGE_PREPROCESS_GEOMETRY_H

#include "image_preprocess_spec.h"

namespace vision_operators {

struct FitResizeDimensions {
    int width = 0;
    int height = 0;
};

struct ImagePreprocessGeometry {
    // Continuous image-edge coordinates: [src_x, src_x + src_width) maps
    // to [dst_x, dst_x + dst_width). Pixel-center sampling is separate;
    // box and point transforms must not add half-pixel offsets.
    int original_width = 0;
    int original_height = 0;
    int output_width = 0;
    int output_height = 0;
    float src_x = 0.0F;
    float src_y = 0.0F;
    float src_width = 0.0F;
    float src_height = 0.0F;
    int dst_x = 0;
    int dst_y = 0;
    int dst_width = 0;
    int dst_height = 0;

    // Validate once before restoring box edges or keypoints.
    void validate() const;
    float restore_x(float x) const noexcept {
        return (x - dst_x) * (src_width / dst_width) + src_x;
    }
    float restore_y(float y) const noexcept {
        return (y - dst_y) * (src_height / dst_height) + src_y;
    }
};

FitResizeDimensions calculate_fit_resize_dimensions(
    float source_width,
    float source_height,
    int output_width,
    int output_height,
    PreprocessResizeRounding rounding =
        PreprocessResizeRounding::kRound);

ImagePreprocessGeometry make_image_preprocess_geometry(
    const ImagePreprocessSpec& spec,
    int input_width,
    int input_height);

}  // namespace vision_operators

#endif  // IMAGE_PREPROCESS_GEOMETRY_H
