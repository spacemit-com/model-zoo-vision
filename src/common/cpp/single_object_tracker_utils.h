/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SINGLE_OBJECT_TRACKER_UTILS_H
#define SINGLE_OBJECT_TRACKER_UTILS_H

#include <array>
#include <vector>

#include <opencv2/core.hpp>

#include "vision_service.h"

namespace vision_deploy {

struct TrackingTensor {
    std::vector<float> values;
    float resize_factor = 0.0f;
};

cv::Rect2f tracking_xyxy_to_xywh(
    const vision::BoundingBox& box,
    const cv::Size& image_size);

vision::BoundingBox tracking_xywh_to_clipped_xyxy(
    const cv::Rect2f& box,
    const cv::Size& image_size,
    float minimum_size);

TrackingTensor preprocess_tracking_patch(
    const cv::Mat& bgr,
    const cv::Rect2f& target,
    float search_area_factor,
    int output_size,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& standard_deviation);

}  // namespace vision_deploy

#endif  // SINGLE_OBJECT_TRACKER_UTILS_H
