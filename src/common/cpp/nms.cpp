/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nms.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace vision_common {

float calculate_iou(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    float x1_inter = std::max(box1.x, box2.x);
    float y1_inter = std::max(box1.y, box2.y);
    float x2_inter = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2_inter = std::min(box1.y + box1.height, box2.y + box2.height);

    // Keep IoU behavior aligned with Python NMS implementation (inclusive coordinates).
    float width_inter = std::max(0.0f, x2_inter - x1_inter + 1.0f);
    float height_inter = std::max(0.0f, y2_inter - y1_inter + 1.0f);
    float area_inter = width_inter * height_inter;

    float area1 = (box1.width + 1.0f) * (box1.height + 1.0f);
    float area2 = (box2.width + 1.0f) * (box2.height + 1.0f);
    float area_union = area1 + area2 - area_inter;

    if (area_union == 0) {
        return 0.0f;
    }
    return area_inter / area_union;
}

float calculate_iou_detection(float x1_1, float y1_1, float x2_1, float y2_1,
                            float x1_2, float y1_2, float x2_2, float y2_2) {
    float x1_inter = std::max(x1_1, x1_2);
    float y1_inter = std::max(y1_1, y1_2);
    float x2_inter = std::min(x2_1, x2_2);
    float y2_inter = std::min(y2_1, y2_2);

    float width_inter = std::max(0.0f, x2_inter - x1_inter + 1.0f);
    float height_inter = std::max(0.0f, y2_inter - y1_inter + 1.0f);
    float area_inter = width_inter * height_inter;

    float area1 = (x2_1 - x1_1 + 1.0f) * (y2_1 - y1_1 + 1.0f);
    float area2 = (x2_2 - x1_2 + 1.0f) * (y2_2 - y1_2 + 1.0f);
    float area_union = area1 + area2 - area_inter;

    if (area_union == 0) {
        return 0.0f;
    }
    return area_inter / area_union;
}

std::vector<int> nms(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float iou_threshold) {
    if (boxes.empty()) {
        return std::vector<int>();
    }

    // Create indices and sort by score
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
                [&scores](int i, int j) { return scores[i] > scores[j]; });

    std::vector<int> keep;
    while (!indices.empty()) {
        int current = indices[0];
        keep.push_back(current);

        if (indices.size() == 1) {
            break;
        }

        std::vector<int> new_indices;
        for (size_t i = 1; i < indices.size(); ++i) {
            float iou = calculate_iou(boxes[current], boxes[indices[i]]);
            if (iou < iou_threshold) {
                new_indices.push_back(indices[i]);
            }
        }
        indices = new_indices;
    }

    return keep;
}

std::vector<Result> multi_class_nms(
    const std::vector<Result>& objects,
    float iou_threshold) {
    if (objects.empty()) {
        return std::vector<Result>();
    }

    // Group by class - find unique labels
    std::vector<int> unique_labels;
    for (const auto& result : objects) {
        if (std::find(unique_labels.begin(), unique_labels.end(), result.label) == unique_labels.end()) {
            unique_labels.push_back(result.label);
        }
    }

    std::vector<Result> final_results;
    for (int label : unique_labels) {
        // Collect results for this class
        std::vector<Result> results_class;
        for (const auto& result : objects) {
            if (result.label == label) {
                results_class.push_back(result);
            }
        }

        // Convert to cv::Rect2f and scores for common nms function
        std::vector<cv::Rect2f> boxes;
        std::vector<float> scores;
        for (const auto& result : results_class) {
            boxes.emplace_back(result.x1, result.y1, result.x2 - result.x1, result.y2 - result.y1);
            scores.push_back(result.score);
        }

        // Use common nms function
        std::vector<int> keep_indices = nms(boxes, scores, iou_threshold);

        // Build final results from kept indices
        for (int idx : keep_indices) {
            final_results.push_back(results_class[idx]);
        }
    }

    return final_results;
}

}  // namespace vision_common

