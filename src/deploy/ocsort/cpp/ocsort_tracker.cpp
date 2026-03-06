/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ocsort_tracker.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> OCSortTracker::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for OCSortTracker");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for OCSortTracker");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    float det_thresh = vision_core::yaml_utils::getFloat(default_params, "det_thresh", 0.3f);
    int max_age = vision_core::yaml_utils::getInt(default_params, "max_age", 30);
    int min_hits = vision_core::yaml_utils::getInt(default_params, "min_hits", 3);
    int delta_t = vision_core::yaml_utils::getInt(default_params, "delta_t", 3);
    float inertia = vision_core::yaml_utils::getFloat(default_params, "inertia", 0.2f);
    bool use_byte = vision_core::yaml_utils::getBool(default_params, "use_byte", false);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<OCSortTracker>(
        model_path,
        conf_threshold,
        iou_threshold,
        det_thresh,
        max_age,
        min_hits,
        delta_t,
        inertia,
        use_byte,
        num_threads,
        lazy_load,
        provider);
}

OCSortTracker::OCSortTracker(const std::string& model_path,
                            float conf_threshold,
                            float iou_threshold,
                            float det_thresh,
                            int max_age,
                            int min_hits,
                            int delta_t,
                            float inertia,
                            bool use_byte,
                            int num_threads,
                            bool lazy_load,
                            const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        det_thresh_(det_thresh),
        max_age_(max_age),
        min_hits_(min_hits),
        delta_t_(delta_t),
        inertia_(inertia),
        use_byte_(use_byte),
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

void OCSortTracker::load_model() {
    // Load detection model
    detector_->load_model();

    // Initialize OC-SORT tracker after detector is loaded
    tracker_ = std::make_unique<ocsort::OCSort>(
        det_thresh_, max_age_, min_hits_, iou_threshold_, delta_t_,
        "iou", inertia_, use_byte_);

    // Copy input shape from detector
    input_shape_ = detector_->get_input_shape();
    model_loaded_ = true;
}

cv::Mat OCSortTracker::preprocess(const cv::Mat& image) {
    ensure_model_loaded();
    return detector_->preprocess(image);
}

std::vector<vision_common::Result> OCSortTracker::track(const cv::Mat& image) {
    ensure_model_loaded();

    // Run detection
    std::vector<vision_common::Result> detections = detector_->detect(image);
    // Convert to Eigen matrix format for OC-SORT tracker
    Eigen::MatrixXf dets = convert_results_to_dets(detections);

    // Update tracker
    std::vector<Eigen::RowVectorXf> tracks = tracker_->update(dets);

    // Convert back to Result format with track_id and preserve label information
    return convert_tracks_to_results(tracks, detections);
}

std::vector<vision_core::ModelCapability> OCSortTracker::get_capabilities() const {
    return {
        vision_core::ModelCapability::kImageInput,
        vision_core::ModelCapability::kTrackUpdate,
        vision_core::ModelCapability::kDraw};
}

Eigen::MatrixXf OCSortTracker::convert_results_to_dets(
    const std::vector<vision_common::Result>& results) {
    if (results.empty()) {
        return Eigen::MatrixXf(0, 6);
    }

    // Each row: [x1, y1, x2, y2, score, label]
    Eigen::MatrixXf dets(results.size(), 6);

    for (size_t i = 0; i < results.size(); ++i) {
        dets(i, 0) = results[i].x1;
        dets(i, 1) = results[i].y1;
        dets(i, 2) = results[i].x2;
        dets(i, 3) = results[i].y2;
        dets(i, 4) = results[i].score;
        dets(i, 5) = static_cast<float>(results[i].label);
    }

    return dets;
}

float OCSortTracker::calculate_iou(float x1_1, float y1_1, float x2_1, float y2_1,
                                    float x1_2, float y1_2, float x2_2,
                                    float y2_2) {
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
}

std::vector<vision_common::Result> OCSortTracker::convert_tracks_to_results(
    const std::vector<Eigen::RowVectorXf>& tracks,
    const std::vector<vision_common::Result>& detections) {

    std::vector<vision_common::Result> results;
    results.reserve(tracks.size());

    for (const auto& track : tracks) {
        vision_common::Result result;

        // OC-SORT output format: [x1, y1, x2, y2, track_id, ...]
        // Note: The exact format depends on OC-SORT implementation
        if (track.size() >= 5) {
            result.x1 = track(0);
            result.y1 = track(1);
            result.x2 = track(2);
            result.y2 = track(3);
            result.track_id = static_cast<int>(track(4));

            // Default score (OC-SORT may not preserve score)
            result.score = 1.0f;

            // Match with detections to find label using IoU
            float max_iou = 0.0f;
            int best_match_idx = -1;

            for (size_t i = 0; i < detections.size(); ++i) {
                float iou = calculate_iou(
                    result.x1, result.y1, result.x2, result.y2,
                    detections[i].x1, detections[i].y1, detections[i].x2,
                    detections[i].y2);
                if (iou > max_iou && iou > 0.5f) {
                    max_iou = iou;
                    best_match_idx = static_cast<int>(i);
                }
            }

            // Use label from matched detection, or -1 if no good match
            if (best_match_idx >= 0) {
                result.label = detections[best_match_idx].label;
                result.score = detections[best_match_idx].score;
            } else {
                result.label = -1;
            }

            results.push_back(result);
        }
    }

    return results;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<OCSortTracker> registrar("OCSortTracker");

}  // namespace vision_deploy
