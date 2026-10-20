/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ocsort_tracker.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {
namespace {

struct OCSortConfig {
    float track_thresh;
    float track_iou_thresh;
    int track_buffer;
    int min_hits;
    int delta_t;
    float inertia;
    bool use_byte;
    std::string asso_func;

    static OCSortConfig from_yaml(const YAML::Node& params) {
        using vision_core::yaml_utils::getBool;
        using vision_core::yaml_utils::getFloat;
        using vision_core::yaml_utils::getInt;
        using vision_core::yaml_utils::getString;
        // Canonical example keys take precedence; old names remain accepted.
        return {
            getFloat(
                params,
                params && params["track_thresh"] ? "track_thresh" : "det_thresh",
                0.3f),
            getFloat(params, "track_iou_thresh", 0.3f),
            getInt(
                params,
                params && params["track_buffer"] ? "track_buffer" : "max_age",
                30),
            getInt(params, "min_hits", 3),
            getInt(params, "delta_t", 3),
            getFloat(params, "inertia", 0.2f),
            getBool(params, "use_byte", false),
            getString(params, "asso_func", "iou"),
        };
    }
};

}  // namespace

std::unique_ptr<vision_core::BaseModel> OCSortTracker::create(const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(config, {
        "conf_threshold", "iou_threshold", "track_thresh", "det_thresh", "track_iou_thresh",
        "track_buffer", "max_age", "min_hits", "delta_t", "inertia", "use_byte", "asso_func"});
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
    const OCSortConfig tracking = OCSortConfig::from_yaml(default_params);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<OCSortTracker>(
        model_path,
        conf_threshold,
        iou_threshold,
        tracking.track_thresh,
        tracking.track_buffer,
        tracking.min_hits,
        tracking.delta_t,
        tracking.inertia,
        tracking.use_byte,
        num_threads,
        lazy_load,
        provider,
        tracking.track_iou_thresh,
        tracking.asso_func);
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
                            const std::string& provider,
                            float track_iou_threshold,
                            const std::string& association_function)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        num_threads_(num_threads),
        provider_(provider) {
    // Validate tracker settings before acquiring detector/model resources.
    tracker_ = std::make_unique<ocsort::OCSort>(
        det_thresh, max_age, min_hits, track_iou_threshold, delta_t,
        association_function, inertia, use_byte);
    // Initialize detector
    detector_ = std::make_unique<YOLOv8Detector>(
        model_path, conf_threshold_, iou_threshold_, num_threads_, lazy_load, provider_);

    // Initialize tracker (will be fully initialized after detector loads)
    if (!lazy_load) {
        load_model();
    }
}

void OCSortTracker::load_model() {
    if (model_loaded_) {
        return;
    }
    // Load detection model (idempotent if detector was eager-loaded in ctor).
    detector_->load_model();

    // Copy input shape from detector
    input_shape_ = detector_->get_input_shape();
    model_loaded_ = true;
}

cv::Mat OCSortTracker::preprocess(const cv::Mat& image) {
    ensure_model_loaded();
    return detector_->preprocess(image);
}

vision_common::TrackingResultList OCSortTracker::track(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold) {
    vision_core::ImageInput input;
    input.image = image;
    return track_input(input, conf_threshold, iou_threshold);
}

vision_common::TrackingResultList OCSortTracker::track_input(
    const vision_core::ImageInput& input,
    float conf_threshold,
    float iou_threshold) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    // Run detection
    const auto t_det0 = std::chrono::steady_clock::now();
    vision_common::DetectionResultList detections =
        detector_->detect_input(
            input, conf_threshold, iou_threshold);
    const auto t_det1 = std::chrono::steady_clock::now();
    // detector_->detect() resets and completes the child instance's profile.
    // Snapshot it immediately: detect_ms below covers the full detector
    // pipeline, while detector.infer is pure ONNX execution.
    const vision_core::RuntimeProfile detector_profile =
        detector_->get_runtime_profile();
    set_runtime_detect_ms(std::chrono::duration<double, std::milli>(t_det1 - t_det0).count());
    set_runtime_preprocess_ms(detector_profile.preprocess_ms);
    set_runtime_model_infer_ms(detector_profile.model_infer_ms);
    set_runtime_postprocess_ms(detector_profile.postprocess_ms);
    for (const auto& component :
        detector_profile.components) {
        add_runtime_component_timing(
            component.name,
            component.total_ms,
            component.calls);
    }
    add_runtime_component_timing(
        "detector.infer", detector_profile.model_infer_ms);

    // Convert to Eigen matrix format for OC-SORT tracker
    const auto t_track0 = std::chrono::steady_clock::now();
    Eigen::MatrixXf dets = convert_results_to_dets(detections);

    // Update tracker
    std::vector<Eigen::RowVectorXf> tracks = tracker_->update(dets);

    // Convert back to TrackingResult format with track_id and preserve label information
    vision_common::TrackingResultList results = convert_tracks_to_results(tracks);
    const auto t_track1 = std::chrono::steady_clock::now();
    set_runtime_track_ms(std::chrono::duration<double, std::milli>(t_track1 - t_track0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


std::vector<vision_core::InferIntent> OCSortTracker::supported_intents() const {
    return {vision_core::InferIntent::kTrack};
}

vision_core::InferResponse OCSortTracker::Run(const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kTrack) {
        return unsupported_intent_response(request.intent);
    }
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "OCSortTracker expects ImageInput";
        return response;
    }

    vision_common::TrackingResultList task_results =
        track_input(
            *image_input,
            request.params.conf_threshold,
            request.params.iou_threshold);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> OCSortTracker::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

void OCSortTracker::configure_preprocess_backend(
    const std::string& backend)
{
    detector_->configure_preprocess_backend(backend);
}

void OCSortTracker::configure_preprocess_fallback(const std::string& fallback)
{
    detector_->configure_preprocess_fallback(fallback);
}

void OCSortTracker::configure_preprocess_opencl_sampling(
    const std::string& sampling)
{
    detector_->configure_preprocess_opencl_sampling(sampling);
}

Eigen::MatrixXf OCSortTracker::convert_results_to_dets(
    const vision_common::DetectionResultList& results) {
    if (results.empty()) {
        return Eigen::MatrixXf(0, 6);
    }

    // Each row: [x1, y1, x2, y2, score, label]
    Eigen::MatrixXf dets(results.size(), 6);

    Eigen::Index count = 0;
    for (const auto& result : results) {
        const auto& box = result.bbox;
        // Clipping an off-image detection can collapse either dimension.
        // Drop unusable detections without failing the entire video frame.
        if (!std::isfinite(box.x1) || !std::isfinite(box.y1) ||
            !std::isfinite(box.x2) || !std::isfinite(box.y2) ||
            !std::isfinite(result.score) ||
            box.x2 <= box.x1 || box.y2 <= box.y1) {
            continue;
        }
        dets.row(count++) << box.x1, box.y1, box.x2, box.y2,
            result.score, static_cast<float>(result.label);
    }
    // Preserve [0,6] when every detection was filtered: update() must still
    // advance the missing-frame state and expire old tracks.
    dets.conservativeResize(count, Eigen::NoChange);

    return dets;
}

vision_common::TrackingResultList OCSortTracker::convert_tracks_to_results(
    const std::vector<Eigen::RowVectorXf>& tracks) {
    vision_common::TrackingResultList results;
    results.reserve(tracks.size());
    for (const auto& track : tracks) {
        if (track.size() != 7) {
            throw std::runtime_error("OC-SORT returned an invalid tracking row");
        }
        vision_common::TrackingResult result;
        result.bbox = {track[0], track[1], track[2], track[3]};
        result.track_id = static_cast<int>(track[4]);
        result.label = static_cast<int>(track[5]);
        result.score = track[6];
        result.state = vision_common::TrackingResult::State::Confirmed;
        results.push_back(result);
    }
    return results;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<OCSortTracker> registrar("OCSortTracker");

}  // namespace vision_deploy
