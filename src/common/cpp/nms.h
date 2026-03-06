/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NMS_H
#define NMS_H

#include <vector>

#include <opencv2/opencv.hpp>

#include "datatype.h"

namespace vision_common {

/**
 * @brief Calculate IoU between two boxes
 * @param box1 First box
 * @param box2 Second box
 * @return IoU value
 */
float calculate_iou(const cv::Rect2f& box1, const cv::Rect2f& box2);

/**
 * @brief Calculate IoU between two detection boxes
 * @param x1_1, y1_1, x2_1, y2_1 Coordinates of first box
 * @param x1_2, y1_2, x2_2, y2_2 Coordinates of second box
 * @return IoU value
 */
float calculate_iou_detection(float x1_1, float y1_1, float x2_1, float y2_1,
                            float x1_2, float y1_2, float x2_2, float y2_2);

/**
 * @brief Non-maximum suppression
 * @param boxes Vector of boxes
 * @param scores Vector of scores
 * @param iou_threshold IoU threshold
 * @return Indices of boxes to keep
 */
std::vector<int> nms(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float iou_threshold
);

/**
 * @brief Multi-class non-maximum suppression
 * @param objects Vector of Result objects with different classes
 * @param iou_threshold IoU threshold for NMS
 * @return Filtered results after per-class NMS
 */
std::vector<Result> multi_class_nms(
    const std::vector<Result>& objects,
    float iou_threshold
);

}  // namespace vision_common

#endif  // NMS_H

