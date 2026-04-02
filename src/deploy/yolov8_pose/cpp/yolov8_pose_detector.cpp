/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov8_pose_detector.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> YOLOv8PoseDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv8PoseDetector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv8PoseDetector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    float point_confidence_threshold =
        vision_core::yaml_utils::getFloat(default_params, "point_confidence_threshold", 0.2f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv8PoseDetector>(
        model_path,
        conf_threshold,
        iou_threshold,
        point_confidence_threshold,
        num_threads,
        lazy_load,
        provider);
}

YOLOv8PoseDetector::YOLOv8PoseDetector(const std::string& model_path,
                                        float conf_threshold,
                                        float iou_threshold,
                                        float point_confidence_threshold,
                                        int num_threads,
                                        bool lazy_load,
                                        const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        point_confidence_threshold_(point_confidence_threshold),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void YOLOv8PoseDetector::load_model() {
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat YOLOv8PoseDetector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    // Use common letterbox function (similar to yolov8_detector.cpp)
    cv::Mat padded = vision_common::letterbox(image,
                                            std::make_pair(inputHeight, inputWidth),
                                            cv::Scalar(0, 0, 0));

    return cv::dnn::blobFromImage(padded, 1.0/255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}



std::vector<vision_common::Result> YOLOv8PoseDetector::estimate_pose(const cv::Mat& image) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    cv::Size orig_size = image.size();

    // Preprocess
    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat inputTensor = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    // Run inference using base class method
    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(inputTensor);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    // Postprocess
    const auto t_post0 = std::chrono::steady_clock::now();
    std::vector<vision_common::Result> results = postprocess(outputs, orig_size);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}

std::vector<vision_core::ModelCapability> YOLOv8PoseDetector::get_capabilities() const {
    return {
        vision_core::ModelCapability::kImageInput,
        vision_core::ModelCapability::kDraw};
}

std::vector<vision_common::Result> YOLOv8PoseDetector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size) {

    // Get output data
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    auto dets_tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dets_dims = dets_tensor_info.GetShape();

    int offset = static_cast<int>(dets_dims[1]);  // 56
    int anchors = static_cast<int>(dets_dims[2]);  // 2100

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    float ratio = std::min(
        static_cast<float>(inputWidth) / static_cast<float>(orig_size.width),
        static_cast<float>(inputHeight) / static_cast<float>(orig_size.height));
    int unpad_w = static_cast<int>(std::round(orig_size.width * ratio));
    int unpad_h = static_cast<int>(std::round(orig_size.height * ratio));

    float dw = (inputWidth - unpad_w) / 2.0f;
    float dh = (inputHeight - unpad_h) / 2.0f;

    // Temporary structure to store detection with source keypoints (before decoding)
    struct DetectionWithSource {
        vision_common::Result result;
        std::vector<float> source_keypoints;  // Raw keypoint data (51 values: 17 * 3)
        int anchor_idx;  // Store anchor index for keypoint decoding
    };

    // First pass: extract all detections with confidence > threshold
    // Store source keypoints data before decoding (as in reference code)
    std::vector<DetectionWithSource> objects_with_source;
    for (int j = 0; j < anchors; ++j) {
        if (output_data[4 * anchors + j] > conf_threshold_) {
            // Decode box
            float half_width = output_data[2 * anchors + j] / 2;
            float half_height = output_data[3 * anchors + j] / 2;
            float x1 = (output_data[j] - half_width - dw) / ratio;
            x1 = std::max(0.0f, x1);
            float y1 = (output_data[anchors + j] - half_height - dh) / ratio;
            y1 = std::max(0.0f, y1);
            float x2 = (output_data[j] + half_width - dw) / ratio;
            x2 = std::max(0.0f, x2);
            float y2 = (output_data[anchors + j] + half_height - dh) / ratio;
            y2 = std::max(0.0f, y2);

            DetectionWithSource det;
            det.result.x1 = x1;
            det.result.y1 = y1;
            det.result.x2 = x2;
            det.result.y2 = y2;
            det.result.score = output_data[4 * anchors + j];
            det.result.label = 0;  // Person class for pose detection
            det.anchor_idx = j;

            // Store source keypoints data (51 values: k=5 to 55)
            det.source_keypoints.clear();
            for (int k = 5; k < 56; ++k) {
                det.source_keypoints.push_back(output_data[k * anchors + j]);
            }

            objects_with_source.push_back(det);
        }
    }

    // Convert to Result for NMS (only using bounding box info)
    std::vector<vision_common::Result> objects_for_nms;
    for (const auto& det : objects_with_source) {
        objects_for_nms.push_back(det.result);
    }

    // Apply NMS on bounding boxes first (as in reference code)
    std::vector<vision_common::Result> objects_nms = nms(objects_for_nms);

    // Second pass: decode keypoints for NMS-filtered detections only
    // Match NMS results back to source data by anchor index
    // We need to track which objects survived NMS
    std::vector<vision_common::Result> final_results;

    // Create a map from Result to anchor index for efficient lookup
    // Since NMS may change order, we match by finding the closest box
    for (const auto& nms_result : objects_nms) {
        // Find the best matching detection in objects_with_source
        int best_match_idx = -1;
        float best_match_score = -1.0f;

        for (size_t i = 0; i < objects_with_source.size(); ++i) {
            const auto& det_with_source = objects_with_source[i];
            // Match by bounding box and score (with tolerance for floating point)
            if (std::abs(det_with_source.result.x1 - nms_result.x1) < 1.0f &&
                std::abs(det_with_source.result.y1 - nms_result.y1) < 1.0f &&
                std::abs(det_with_source.result.x2 - nms_result.x2) < 1.0f &&
                std::abs(det_with_source.result.y2 - nms_result.y2) < 1.0f &&
                std::abs(det_with_source.result.score - nms_result.score) < 0.01f) {
                // Use score as tie-breaker for best match
                if (det_with_source.result.score > best_match_score) {
                    best_match_idx = static_cast<int>(i);
                    best_match_score = det_with_source.result.score;
                }
            }
        }

        if (best_match_idx >= 0) {
            // Decode keypoints for this detection (as in reference code)
            const auto& det_with_source = objects_with_source[best_match_idx];
            vision_common::Result final_result = nms_result;
            final_result.keypoints.clear();

            // Decode keypoints: k=0 to 16, each has 3 values (x, y, visibility)
            // source_keypoints has 51 values: indices 0-50 correspond to k=5-55 in output
            // So k-th keypoint is at indices [k*3, k*3+1, k*3+2] in source_keypoints
            for (int k = 0; k < 17; ++k) {
                vision_common::KeyPoint kp;
                kp.x = (det_with_source.source_keypoints[k * 3] - dw) / ratio;
                kp.y = (det_with_source.source_keypoints[k * 3 + 1] - dh) / ratio;
                kp.visibility = det_with_source.source_keypoints[k * 3 + 2];
                final_result.keypoints.push_back(kp);
            }
            final_results.push_back(final_result);
        }
    }

    return final_results;
}

float YOLOv8PoseDetector::calculate_iou(const vision_common::Result& det1, const vision_common::Result& det2) {
    // Use common calculate_iou_detection function
    return vision_common::calculate_iou_detection(
        det1.x1, det1.y1, det1.x2, det1.y2,
        det2.x1, det2.y1, det2.x2, det2.y2);
}

std::vector<vision_common::Result> YOLOv8PoseDetector::nms(const std::vector<vision_common::Result>& dets) {
    if (dets.empty()) {
        return std::vector<vision_common::Result>();
    }

    // Convert Result to cv::Rect2f and scores for common nms function
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    boxes.reserve(dets.size());
    scores.reserve(dets.size());

    for (const auto& det : dets) {
        boxes.push_back(cv::Rect2f(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1));
        scores.push_back(det.score);
    }

    // Use common nms function
    std::vector<int> keep_indices = vision_common::nms(boxes, scores, iou_threshold_);

    // Convert back to Result vector
    std::vector<vision_common::Result> final_dets;
    final_dets.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        final_dets.push_back(dets[idx]);
    }

    return final_dets;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv8PoseDetector> registrar("YOLOv8PoseDetector");

}  // namespace vision_deploy

