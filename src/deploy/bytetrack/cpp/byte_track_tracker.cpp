/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "byte_track_tracker.h"

#include <chrono>
#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> ByteTrackTracker::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for ByteTrackTracker");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for ByteTrackTracker");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int frame_rate = vision_core::yaml_utils::getInt(default_params, "frame_rate", 30);
    int track_buffer = vision_core::yaml_utils::getInt(default_params, "track_buffer", 30);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<ByteTrackTracker>(
        model_path,
        conf_threshold,
        iou_threshold,
        frame_rate,
        track_buffer,
        num_threads,
        lazy_load,
        provider);
}

ByteTrackTracker::ByteTrackTracker(const std::string& model_path,
                                    float conf_threshold,
                                    float iou_threshold,
                                    int frame_rate,
                                    int track_buffer,
                                    int num_threads,
                                    bool lazy_load,
                                    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        frame_rate_(frame_rate),
        track_buffer_(track_buffer),
        num_threads_(num_threads),
        provider_(provider) {
    // Initialize detector
    detector_ = std::make_unique<YOLOv8Detector>(
        model_path, conf_threshold_, iou_threshold_, num_threads_, lazy_load, provider_);

    // Initialize tracker (will be fully initialized after detector loads)
    if (!lazy_load) {
        load_model();
    }
}

void ByteTrackTracker::load_model() {
    // Load detection model
    detector_->load_model();

    // Initialize tracker after detector is loaded
    tracker_ = std::make_unique<BYTETracker>(frame_rate_, track_buffer_);

    // Copy input shape from detector
    input_shape_ = detector_->get_input_shape();
    model_loaded_ = true;
}

cv::Mat ByteTrackTracker::preprocess(const cv::Mat& image) {
    ensure_model_loaded();
    return detector_->preprocess(image);
}

std::vector<vision_common::Result> ByteTrackTracker::track(const cv::Mat& image) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    // Run detection
    const auto t_det0 = std::chrono::steady_clock::now();
    std::vector<vision_common::Result> detections = detector_->detect(image);
    const auto t_det1 = std::chrono::steady_clock::now();
    set_runtime_detect_ms(std::chrono::duration<double, std::milli>(t_det1 - t_det0).count());

    // Convert to Object format for tracker
    const auto t_track0 = std::chrono::steady_clock::now();
    std::vector<Object> objects = convert_results_to_objects(detections);

    // Update tracker
    std::vector<STrack> stracks = tracker_->update(objects);

    // Convert back to Result format with track_id and preserve label information
    std::vector<vision_common::Result> results = convert_stracks_to_results(stracks, detections);
    const auto t_track1 = std::chrono::steady_clock::now();
    set_runtime_track_ms(std::chrono::duration<double, std::milli>(t_track1 - t_track0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}

std::vector<vision_core::ModelCapability> ByteTrackTracker::get_capabilities() const {
    return {
        vision_core::ModelCapability::kImageInput,
        vision_core::ModelCapability::kTrackUpdate,
        vision_core::ModelCapability::kDraw};
}

std::vector<Object> ByteTrackTracker::convert_results_to_objects(
    const std::vector<vision_common::Result>& results) {
    std::vector<Object> objects;
    objects.reserve(results.size());

    for (const auto& result : results) {
        Object obj;
        obj.rect.x = result.x1;
        obj.rect.y = result.y1;
        obj.rect.width = result.x2 - result.x1;
        obj.rect.height = result.y2 - result.y1;
        obj.label = result.label;
        obj.prob = result.score;
        objects.push_back(obj);
    }

    return objects;
}

std::vector<vision_common::Result> ByteTrackTracker::convert_stracks_to_results(
    const std::vector<STrack>& stracks,
    const std::vector<vision_common::Result>& detections) {
    std::vector<vision_common::Result> results;
    results.reserve(stracks.size());

    // Helper function to calculate IoU
    auto calculate_iou = [](float x1_1, float y1_1, float x2_1, float y2_1,
                            float x1_2, float y1_2, float x2_2, float y2_2) -> float {
        float inter_x1 = std::max(x1_1, x1_2);
        float inter_y1 = std::max(y1_1, y1_2);
        float inter_x2 = std::min(x2_1, x2_2);
        float inter_y2 = std::min(y2_1, y2_2);

        if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) {
            return 0.0f;
        }

        float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
        float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
        float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
        float union_area = area1 + area2 - inter_area;

        if (union_area <= 0.0f) {
            return 0.0f;
        }

        return inter_area / union_area;
    };

    for (const auto& strack : stracks) {
        vision_common::Result result;

        // Convert tlwh to xyxy
        float track_x1 = strack.tlwh[0];
        float track_y1 = strack.tlwh[1];
        float track_x2 = strack.tlwh[0] + strack.tlwh[2];
        float track_y2 = strack.tlwh[1] + strack.tlwh[3];

        result.x1 = track_x1;
        result.y1 = track_y1;
        result.x2 = track_x2;
        result.y2 = track_y2;

        result.score = strack.score;
        result.track_id = strack.track_id;

        // Match with detections to find label
        // Find the detection with highest IoU
        float max_iou = 0.0f;
        int best_match_idx = -1;

        for (size_t i = 0; i < detections.size(); i++) {
            float iou = calculate_iou(
                track_x1, track_y1, track_x2, track_y2,
                detections[i].x1, detections[i].y1, detections[i].x2,
                detections[i].y2);

            if (iou > max_iou && iou > 0.5f) {  // Threshold to ensure good match
                max_iou = iou;
                best_match_idx = i;
            }
        }

        // Use label from matched detection, or -1 if no good match
        if (best_match_idx >= 0) {
            result.label = detections[best_match_idx].label;
        } else {
            result.label = -1;
        }

        results.push_back(result);
    }

    return results;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<ByteTrackTracker> registrar("ByteTrackTracker");

}  // namespace vision_deploy

