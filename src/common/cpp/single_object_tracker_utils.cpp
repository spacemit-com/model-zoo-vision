/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "single_object_tracker_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace vision_deploy {

namespace {

bool finite_box(const cv::Rect2f& box) {
    return std::isfinite(box.x) && std::isfinite(box.y) &&
        std::isfinite(box.width) && std::isfinite(box.height);
}

}  // namespace

cv::Rect2f tracking_xyxy_to_xywh(
    const vision::BoundingBox& box,
    const cv::Size& image_size) {
    if (image_size.width <= 0 || image_size.height <= 0) {
        throw std::invalid_argument(
            "tracking image dimensions must be positive");
    }
    const cv::Rect2f converted(
        box.x1,
        box.y1,
        box.x2 - box.x1,
        box.y2 - box.y1);
    if (!finite_box(converted) ||
        converted.width <= 0.0f || converted.height <= 0.0f) {
        throw std::invalid_argument(
            "tracking initial box must be finite with positive area");
    }
    if (converted.x < 0.0f || converted.y < 0.0f ||
        box.x2 > image_size.width || box.y2 > image_size.height) {
        throw std::invalid_argument(
            "tracking initial box must be inside the image");
    }
    return converted;
}

vision::BoundingBox tracking_xywh_to_clipped_xyxy(
    const cv::Rect2f& box,
    const cv::Size& image_size,
    float minimum_size) {
    if (image_size.width <= 0 || image_size.height <= 0 ||
        !finite_box(box) || !std::isfinite(minimum_size) ||
        minimum_size <= 0.0f) {
        throw std::invalid_argument(
            "tracking box clipping arguments are invalid");
    }
    const float min_width = std::min(
        minimum_size, static_cast<float>(image_size.width));
    const float min_height = std::min(
        minimum_size, static_cast<float>(image_size.height));
    const float x1 = std::clamp(
        box.x,
        0.0f,
        static_cast<float>(image_size.width) - min_width);
    const float y1 = std::clamp(
        box.y,
        0.0f,
        static_cast<float>(image_size.height) - min_height);
    const float raw_x2 = box.x + std::max(box.width, min_width);
    const float raw_y2 = box.y + std::max(box.height, min_height);
    const float x2 = std::clamp(
        raw_x2,
        x1 + min_width,
        static_cast<float>(image_size.width));
    const float y2 = std::clamp(
        raw_y2,
        y1 + min_height,
        static_cast<float>(image_size.height));
    return {x1, y1, x2, y2};
}

TrackingTensor preprocess_tracking_patch(
    const cv::Mat& bgr,
    const cv::Rect2f& target,
    float search_area_factor,
    int output_size,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& standard_deviation) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument(
            "tracking expects a non-empty BGR8 image");
    }
    if (!finite_box(target) ||
        target.width <= 0.0f || target.height <= 0.0f ||
        !std::isfinite(search_area_factor) ||
        search_area_factor <= 0.0f || output_size <= 0) {
        throw std::invalid_argument(
            "tracking crop arguments are invalid");
    }
    for (size_t channel = 0; channel < standard_deviation.size();
        ++channel) {
        if (!std::isfinite(mean[channel]) ||
            !std::isfinite(standard_deviation[channel]) ||
            standard_deviation[channel] <= 0.0f) {
            throw std::invalid_argument(
                "tracking normalization values are invalid");
        }
    }

    const int crop_size = static_cast<int>(std::ceil(
        std::sqrt(target.width * target.height) *
        search_area_factor));
    if (crop_size < 1) {
        throw std::invalid_argument("tracking crop is empty");
    }
    const int x1 = static_cast<int>(std::round(
        target.x + 0.5f * target.width - 0.5f * crop_size));
    const int y1 = static_cast<int>(std::round(
        target.y + 0.5f * target.height - 0.5f * crop_size));
    const int x2 = x1 + crop_size;
    const int y2 = y1 + crop_size;
    const int pad_left = std::max(0, -x1);
    const int pad_top = std::max(0, -y1);
    const int pad_right = std::max(0, x2 - bgr.cols);
    const int pad_bottom = std::max(0, y2 - bgr.rows);
    const int valid_x1 = x1 + pad_left;
    const int valid_y1 = y1 + pad_top;
    const int valid_x2 = x2 - pad_right;
    const int valid_y2 = y2 - pad_bottom;
    if (valid_x1 >= valid_x2 || valid_y1 >= valid_y2) {
        throw std::invalid_argument(
            "tracking target crop does not intersect the image");
    }

    cv::Mat patch = bgr(
        cv::Range(valid_y1, valid_y2),
        cv::Range(valid_x1, valid_x2)).clone();
    cv::copyMakeBorder(
        patch,
        patch,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv::BORDER_CONSTANT);
    cv::resize(
        patch,
        patch,
        cv::Size(output_size, output_size),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    cv::cvtColor(patch, patch, cv::COLOR_BGR2RGB);
    patch.convertTo(patch, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(patch, channels);
    TrackingTensor output;
    output.values.resize(
        static_cast<size_t>(3) * output_size * output_size);
    const size_t plane =
        static_cast<size_t>(output_size) * output_size;
    for (size_t channel = 0; channel < channels.size(); ++channel) {
        channels[channel] =
            (channels[channel] - mean[channel]) /
            standard_deviation[channel];
        std::memcpy(
            output.values.data() + channel * plane,
            channels[channel].ptr<float>(),
            plane * sizeof(float));
    }
    output.resize_factor =
        static_cast<float>(output_size) / crop_size;
    return output;
}

}  // namespace vision_deploy
