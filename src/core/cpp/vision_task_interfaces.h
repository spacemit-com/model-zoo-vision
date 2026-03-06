/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VISION_TASK_INTERFACES_H
#define VISION_TASK_INTERFACES_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/cpp/datatype.h"

namespace vision_core {

using DetectionResultList = std::vector<vision_common::Result>;
using ClassificationResultList = std::vector<vision_common::Result>;
using SegmentationResultList = std::vector<vision_common::Result>;
using PoseResultList = std::vector<vision_common::Result>;
using TrackingResultList = std::vector<vision_common::Result>;
using EmbeddingVector = std::vector<float>;
using ActionScores = std::vector<float>;

class IDetectionModel {
public:
    virtual ~IDetectionModel() = default;
    virtual DetectionResultList detect(const cv::Mat& image) = 0;
};

class IClassificationModel {
public:
    virtual ~IClassificationModel() = default;
    virtual ClassificationResultList classify(const cv::Mat& image) = 0;
};

class ISegmentationModel {
public:
    virtual ~ISegmentationModel() = default;
    virtual SegmentationResultList segment(const cv::Mat& image) = 0;
};

class IPoseModel {
public:
    virtual ~IPoseModel() = default;
    virtual PoseResultList estimate_pose(const cv::Mat& image) = 0;
};

class ITrackingModel {
public:
    virtual ~ITrackingModel() = default;
    virtual TrackingResultList track(const cv::Mat& image) = 0;
};

class IEmbeddingModel {
public:
    virtual ~IEmbeddingModel() = default;
    virtual EmbeddingVector infer_embedding(const cv::Mat& image) = 0;
};

class ISequenceActionModel {
public:
    virtual ~ISequenceActionModel() = default;
    virtual ActionScores infer_sequence(const float* pts, int image_width, int image_height) = 0;
    /** Optional: class names (e.g. STGCN 7 classes). Default empty. */
    virtual std::vector<std::string> get_class_names() const { return {}; }
    /** Optional: fall-down class index (e.g. STGCN = 6). -1 if N/A. */
    virtual int get_fall_down_class_index() const { return -1; }
};

}  // namespace vision_core

#endif  // VISION_TASK_INTERFACES_H
