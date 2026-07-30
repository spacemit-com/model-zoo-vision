/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <iostream>
#include <string>

#include "operators/image_preprocess/image_preprocess_geometry.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check_close(float actual, float expected, const std::string& message)
{
    check(
        std::fabs(actual - expected) < 0.0001F,
        message + " (expected " + std::to_string(expected) +
            ", got " + std::to_string(actual) + ")");
}

}  // namespace

int main()
{
    using vision_operators::ImagePreprocessSpec;
    using vision_operators::PreprocessCropMode;
    using vision_operators::PreprocessResizeRounding;
    using vision_operators::PreprocessResizeMode;
    using vision_operators::make_image_preprocess_geometry;

    ImagePreprocessSpec stretch_spec;
    stretch_spec.output_width = 640;
    stretch_spec.output_height = 320;
    const auto stretch =
        make_image_preprocess_geometry(stretch_spec, 1280, 720);
    check(stretch.dst_x == 0 && stretch.dst_y == 0,
            "stretch starts at the output origin");
    check(stretch.dst_width == 640 && stretch.dst_height == 320,
            "stretch fills the requested output");
    check_close(stretch.src_width, 1280.0F, "stretch keeps source width");
    check_close(stretch.src_height, 720.0F, "stretch keeps source height");

    ImagePreprocessSpec letterbox_spec;
    letterbox_spec.output_width = 640;
    letterbox_spec.output_height = 640;
    letterbox_spec.resize_mode = PreprocessResizeMode::kLetterbox;
    const auto letterbox =
        make_image_preprocess_geometry(letterbox_spec, 1280, 720);
    check(letterbox.dst_x == 0 && letterbox.dst_y == 140,
            "letterbox centers the resized image");
    check(letterbox.dst_width == 640 && letterbox.dst_height == 360,
            "letterbox preserves the source aspect ratio");

    ImagePreprocessSpec top_left_spec = letterbox_spec;
    top_left_spec.resize_mode = PreprocessResizeMode::kFitTopLeft;
    const auto top_left =
        make_image_preprocess_geometry(top_left_spec, 1280, 720);
    check(top_left.dst_x == 0 && top_left.dst_y == 0,
            "fit-top-left anchors the resized image at the origin");
    check(top_left.dst_width == 640 && top_left.dst_height == 360,
            "fit-top-left preserves the source aspect ratio");

    ImagePreprocessSpec truncate_spec = top_left_spec;
    truncate_spec.resize_rounding =
        PreprocessResizeRounding::kTruncate;
    const auto truncated =
        make_image_preprocess_geometry(
            truncate_spec, 1000, 334);
    check(
        truncated.dst_width == 640 &&
            truncated.dst_height == 213,
        "truncate resize matches SCRFD integer dimensions");

    ImagePreprocessSpec square_spec;
    square_spec.output_width = 224;
    square_spec.output_height = 224;
    square_spec.crop_mode = PreprocessCropMode::kCenterSquare;
    const auto square =
        make_image_preprocess_geometry(square_spec, 1280, 720);
    check_close(square.src_x, 280.0F, "center-square offsets source x");
    check_close(square.src_y, 0.0F, "center-square keeps source y");
    check_close(square.src_width, 720.0F, "center-square source width");
    check_close(square.src_height, 720.0F, "center-square source height");
    ImagePreprocessSpec short_side_spec;
    short_side_spec.output_width = 224;
    short_side_spec.output_height = 224;
    short_side_spec.crop_mode =
        PreprocessCropMode::kResizeShortSideCenterCrop;
    short_side_spec.resize_width = 256;
    short_side_spec.resize_height = 384;
    const auto short_side =
        make_image_preprocess_geometry(short_side_spec, 640, 480);
    check_close(short_side.src_x, 40.0F,
                "resize-short-side center crop offsets source x");
    check_close(short_side.src_y, 100.0F,
                "resize-short-side center crop offsets source y");
    check_close(short_side.src_width, 560.0F,
                "resize-short-side center crop source width");
    check_close(short_side.src_height, 280.0F,
                "resize-short-side center crop source height");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: image preprocess geometry\n";
    return 0;
}
