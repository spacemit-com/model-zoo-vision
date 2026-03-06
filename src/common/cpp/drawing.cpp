/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "drawing.h"

#include <algorithm>
#include <string>
#include <vector>

namespace vision_common {

void draw_detections(
    cv::Mat& image,
    const std::vector<vision_common::Result>& detections,
    const std::vector<std::string>& labels,
    const cv::Scalar& box_color,
    int line_thickness) {
    if (detections.empty()) return;
    for (const auto& det : detections) {
        int ix1 = static_cast<int>(det.x1);
        int iy1 = static_cast<int>(det.y1);
        int ix2 = static_cast<int>(det.x2);
        int iy2 = static_cast<int>(det.y2);

        // Draw bounding box
        cv::rectangle(image, cv::Point(ix1, iy1), cv::Point(ix2, iy2), box_color, line_thickness);

        // Draw label and score (skip if class_id is -1, e.g., for face detection without labels)
        if (det.label >= 0) {
            std::string labelText;
            if (!labels.empty() && det.label < static_cast<int>(labels.size())) {
                labelText = labels[det.label] + ": " + std::to_string(det.score).substr(0, 4);
            } else {
                labelText = "Class " + std::to_string(det.label) + ": " + std::to_string(det.score).substr(0, 4);
            }
            if (det.track_id >= 0) {
                labelText += " ID:" + std::to_string(det.track_id);
            }

            cv::putText(image, labelText, cv::Point(ix1, iy1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, box_color, line_thickness);
        }
    }
}

void draw_keypoints(
    cv::Mat& image,
    const std::vector<vision_common::Result>& results,
    float point_confidence_threshold,
    const cv::Scalar& box_color,
    const cv::Scalar& kp_color,
    int line_thickness,
    int kp_radius) {
    if (results.empty()) return;
    // Keypoint connections for pose estimation (COCO format: 17 keypoints)
    // These connections define the skeleton structure
    std::vector<std::pair<int, int>> kp_connections = {
        {16, 14}, {14, 12}, {15, 13}, {13, 11}, {12, 11},  // Head to shoulders
        {5, 7}, {7, 9}, {6, 8}, {8, 10},                   // Arms
        {5, 6}, {5, 11}, {6, 12},                          // Shoulders to torso
        {11, 13}, {12, 14},                                 // Torso to hips
        {0, 1}, {0, 2}, {1, 3}, {2, 4},                    // Face
        {0, 5}, {0, 6},                                     // Face to shoulders
        {3, 5}, {4, 6}  // Face to shoulders (alternate)
    };

    for (const auto& result : results) {
        int x1 = static_cast<int>(result.x1);
        int y1 = static_cast<int>(result.y1);
        int x2 = static_cast<int>(result.x2);
        int y2 = static_cast<int>(result.y2);

        // Draw bounding box
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), box_color, line_thickness);

        // Draw label and score
        std::string labelText = "Person: " + std::to_string(result.score).substr(0, 4);
        cv::putText(image, labelText, cv::Point(x1, y1 - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, box_color, line_thickness);

        // Draw keypoints
        for (size_t i = 0; i < result.keypoints.size(); ++i) {
            const auto& kp = result.keypoints[i];
            if (kp.visibility < point_confidence_threshold) {
                continue;  // Skip invisible keypoints
            }
            // Draw keypoint circle
            cv::circle(image, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                        kp_radius, kp_color, -1);
        }

        // Draw keypoint connections (skeleton)
        for (const auto& connection : kp_connections) {
            int start_idx = connection.first;
            int end_idx = connection.second;

            // Check bounds
            if (start_idx >= static_cast<int>(result.keypoints.size()) ||
                end_idx >= static_cast<int>(result.keypoints.size())) {
                continue;
            }

            const auto& start_kp = result.keypoints[start_idx];
            const auto& end_kp = result.keypoints[end_idx];

            // Only draw line if both keypoints are visible
            if (start_kp.visibility < point_confidence_threshold ||
                end_kp.visibility < point_confidence_threshold) {
                continue;
            }

            cv::line(image,
                    cv::Point(static_cast<int>(start_kp.x), static_cast<int>(start_kp.y)),
                    cv::Point(static_cast<int>(end_kp.x), static_cast<int>(end_kp.y)),
                    kp_color, line_thickness);
        }
    }
}

void draw_segmentation(
    cv::Mat& image,
    const std::vector<vision_common::Result>& results,
    const std::vector<std::string>& labels,
    float alpha,
    const cv::Scalar& box_color,
    int line_thickness) {
    if (results.empty()) return;
    // Generate random colors for each mask
    std::vector<cv::Scalar> colors;
    cv::RNG rng(12345);
    for (size_t i = 0; i < results.size(); ++i) {
        colors.push_back(cv::Scalar(
            rng.uniform(0, 255),
            rng.uniform(0, 255),
            rng.uniform(0, 255)));
    }

    // Create overlay image for masks only
    cv::Mat overlay = image.clone();

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        cv::Scalar color = colors[i];

        // Draw mask if available
        if (result.mask != nullptr && !result.mask->empty()) {
            cv::Mat mask = *result.mask;

            // Ensure mask is CV_8U type (required by setTo)
            if (mask.type() != CV_8U) {
                cv::Mat mask_uint8;
                if (mask.type() == CV_32F) {
                    // Convert from float [0, 1] to uint8 [0, 255]
                    mask.convertTo(mask_uint8, CV_8U, 255.0);
                } else {
                    mask.convertTo(mask_uint8, CV_8U);
                }
                mask = mask_uint8;
            }

            // Resize mask if needed (must match image size)
            if (mask.size() != image.size()) {
                cv::Mat mask_resized;
                cv::resize(mask, mask_resized, image.size(), 0, 0, cv::INTER_NEAREST);
                mask = mask_resized;
            }

            // Ensure mask is single channel (required by setTo)
            if (mask.channels() != 1) {
                cv::Mat mask_single;
                if (mask.channels() == 3) {
                    cv::cvtColor(mask, mask_single, cv::COLOR_BGR2GRAY);
                } else {
                    // Extract first channel
                    std::vector<cv::Mat> channels;
                    cv::split(mask, channels);
                    mask_single = channels[0];
                }
                mask = mask_single;
            }

            // Apply mask to overlay (mask must be CV_8U, single channel, same size as overlay)
            overlay.setTo(color, mask);
        }
    }

    // Blend overlay with original image, then draw boxes/labels on top at full opacity
    cv::addWeighted(image, 1.0 - alpha, overlay, alpha, 0, image);

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        int x1 = static_cast<int>(result.x1);
        int y1 = static_cast<int>(result.y1);
        int x2 = static_cast<int>(result.x2);
        int y2 = static_cast<int>(result.y2);

        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), box_color, line_thickness);

        if (result.label >= 0) {
            std::string labelText;
            if (!labels.empty() && result.label < static_cast<int>(labels.size())) {
                labelText = labels[result.label] + ": " + std::to_string(result.score).substr(0, 4);
            } else {
                labelText = "Class " + std::to_string(result.label) + ": " + std::to_string(result.score).substr(0, 4);
            }
            if (result.track_id >= 0) {
                labelText += " ID:" + std::to_string(result.track_id);
            }
            int label_y = std::max(2, y1 - 10);
            cv::putText(image, labelText, cv::Point(x1, label_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.9, box_color, line_thickness);
        }
    }
}

/**
 * @brief Get color for track ID
 */
cv::Scalar get_track_color(int track_id) {
    // Use a simple hash to generate consistent colors for each track ID
    static const int colors[][3] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0},
        {255, 0, 255}, {0, 255, 255}, {128, 0, 0}, {0, 128, 0},
        {0, 0, 128}, {128, 128, 0}, {128, 0, 128}, {0, 128, 128}
    };
    constexpr int num_colors = 12;
    int color_idx = (track_id % num_colors + num_colors) % num_colors;
    return cv::Scalar(colors[color_idx][0], colors[color_idx][1], colors[color_idx][2]);
}

void draw_tracking_results(
    cv::Mat& image,
    const std::vector<Result>& results,
    const std::vector<std::string>& labels,
    int line_thickness) {
    if (results.empty()) return;
    for (const auto& result : results) {
        // Get color for this track
        cv::Scalar color = get_track_color(result.track_id);

        // Draw bounding box
        cv::rectangle(image,
                        cv::Point(static_cast<int>(result.x1), static_cast<int>(result.y1)),
                        cv::Point(static_cast<int>(result.x2), static_cast<int>(result.y2)),
                        color, line_thickness);

        // Prepare label text: class ID score (one line)
        std::string label_text;
        if (result.label >= 0 && result.label < static_cast<int>(labels.size())) {
            label_text = labels[result.label] + " ID:" + std::to_string(result.track_id);
        } else {
            label_text = "ID:" + std::to_string(result.track_id);
        }
        label_text += " " + std::to_string(static_cast<int>(result.score * 100)) + "%";

        // Draw label background (clamp to image to avoid negative coordinates)
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int label_y_top = std::max(0, static_cast<int>(result.y1) - text_size.height - 5);
        int label_y_bottom = static_cast<int>(result.y1);
        cv::rectangle(image,
                        cv::Point(static_cast<int>(result.x1), label_y_top),
                        cv::Point(static_cast<int>(result.x1) + text_size.width, label_y_bottom),
                        color, -1);

        // Draw label text
        cv::putText(image, label_text,
                        cv::Point(static_cast<int>(result.x1), std::max(2, static_cast<int>(result.y1) - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

}  // namespace vision_common

