/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DATATYPE_H
#define DATATYPE_H

#include <memory>
#include <vector>

// Forward declaration to avoid including OpenCV headers
namespace cv {
    class Mat;
}

/**
 * @file datatype.h
 * @brief Common data types for computer vision models
 * 
 * This file contains common data structures used across different model types.
 */

namespace vision_common {

/**
 * @brief Keypoint structure for pose estimation models
 */
struct KeyPoint {
    float x;           // X coordinate
    float y;           // Y coordinate
    float visibility;  // Visibility/confidence score (0.0-1.0)
};

/**
 * @brief Unified result structure for all computer vision models
 * 
 * This structure can represent results from:
 * - Object detection: bounding box + class + score
 * - Pose estimation: bounding box + keypoints
 * - Face recognition: embedding vector
 * - Classification: class_scores (all class probabilities)
 * - Segmentation: bounding box + mask (optional)
 */
struct Result {
    // Bounding box coordinates (used by detection, pose, and segmentation models)
    // For classification models, these are typically 0
    float x1, y1, x2, y2;

    // Confidence score (for detection/pose models) or top class score (for classification)
    float score;

    // Class label/ID (for detection models) or predicted class (for classification)
    int label;

    // Keypoints for pose estimation models
    // If empty, this is not a pose estimation result
    std::vector<KeyPoint> keypoints;

    // Embedding vector for face recognition models (e.g., ArcFace)
    // If empty, this is not an embedding result
    std::vector<float> embedding;

    // Class scores for classification models (all class probabilities)
    // If empty, this is not a classification result
    std::vector<float> class_scores;

    // Segmentation mask for segmentation models
    // Using shared_ptr to avoid including OpenCV headers in this header file
    // If nullptr, this is not a segmentation result
    std::shared_ptr<cv::Mat> mask;  // If nullptr, not a segmentation result

    // Track ID for tracking models (e.g., ByteTrack).  -1 if not tracking.
    int track_id = -1;
};

}  // namespace vision_common

#endif  // DATATYPE_H
