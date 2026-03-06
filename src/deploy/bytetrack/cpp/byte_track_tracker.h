/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef BYTE_TRACK_TRACKER_H
#define BYTE_TRACK_TRACKER_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"
#include "include/BYTETracker.h"
#include "yolov8_detector.h"

namespace YAML {
class Node;
}

namespace cv {
class Mat;
}

namespace vision_deploy {

/**
 * @brief ByteTrack Tracker with integrated detection
 *
 * Combines YOLOv8 detection and ByteTrack tracking for video analysis.
 */
class ByteTrackTracker : public vision_core::BaseModel, public vision_core::ITrackingModel {
public:
    ByteTrackTracker(const std::string& model_path,
                    float conf_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    int frame_rate = 30,
                    int track_buffer = 30,
                    int num_threads = 4,
                    bool lazy_load = false,
                    const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~ByteTrackTracker() = default;

    /**
     * @brief Load detection model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for detection
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /**
     * @brief Update tracker with new detections
     * @param image Input image
     * @return Vector of tracking results with track_id
     */
    std::vector<vision_common::Result> track(const cv::Mat& image);

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

private:
    float conf_threshold_;
    float iou_threshold_;
    int frame_rate_;
    int track_buffer_;
    int num_threads_;
    std::string provider_;

    // Detection model
    std::unique_ptr<YOLOv8Detector> detector_;

    // Tracking model
    std::unique_ptr<BYTETracker> tracker_;

    /**
     * @brief Convert Result to Object format for tracker
     */
    std::vector<Object> convert_results_to_objects(const std::vector<vision_common::Result>& results);

    /**
     * @brief Convert STrack to Result format, preserving label from detections
     */
    std::vector<vision_common::Result> convert_stracks_to_results(
        const std::vector<STrack>& stracks,
        const std::vector<vision_common::Result>& detections);
};

}  // namespace vision_deploy

#endif  // BYTE_TRACK_TRACKER_H
