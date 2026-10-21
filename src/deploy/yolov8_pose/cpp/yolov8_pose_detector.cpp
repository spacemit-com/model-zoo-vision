/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov8_pose_detector.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> YOLOv8PoseDetector::create(const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(config, {
        "conf_threshold", "iou_threshold", "kp_threshold", "point_confidence_threshold"});
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
        vision_core::yaml_utils::getAliasedFloat(
            default_params, "kp_threshold", "point_confidence_threshold", 0.2f);
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
    if (!std::isfinite(conf_threshold_) || conf_threshold_ < 0.0f || conf_threshold_ > 1.0f ||
        !std::isfinite(iou_threshold_) || iou_threshold_ < 0.0f || iou_threshold_ > 1.0f ||
        !std::isfinite(point_confidence_threshold_) || point_confidence_threshold_ < 0.0f ||
        point_confidence_threshold_ > 1.0f) {
        throw std::invalid_argument("YOLOv8-Pose thresholds must be finite and in [0,1]");
    }
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

void YOLOv8PoseDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_shape_.size() != 4 || input_shape_[0] > 1 || input_shape_[1] != 3 ||
        input_shape_[2] <= 0 || input_shape_[3] <= 0 ||
        input_shape_[2] > std::numeric_limits<int>::max() ||
        input_shape_[3] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("YOLOv8-Pose expects float32 input [1,3,H,W] with fixed positive H,W");
    }
    model_loaded_ = true;
}

cv::Mat YOLOv8PoseDetector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    return vision_common::letterbox_to_nchw_rgb_blob(
        image,
        std::make_pair(inputHeight, inputWidth),
        cv::Scalar(0, 0, 0));
}



vision_common::PoseResultList
YOLOv8PoseDetector::estimate_pose_input(
    const vision_core::ImageInput& input,
    float conf_threshold,
    float iou_threshold, float kp_threshold, int max_det) {
    const float use_kp = vision_core::resolve_probability_threshold(
        kp_threshold, point_confidence_threshold_, "kp_threshold");
    const float use_conf = vision_core::resolve_probability_threshold(
        conf_threshold, conf_threshold_, "conf_threshold");
    const float use_iou = vision_core::resolve_probability_threshold(
        iou_threshold, iou_threshold_, "iou_threshold");
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const cv::Size orig_size(
        input.image.cols,
        input.format == vision_core::ImagePixelFormat::kNv12
            ? input.image.rows * 2 / 3
            : input.image.rows);

    // Preprocess
    const auto t_pre0 = std::chrono::steady_clock::now();
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = static_cast<int>(input_shape_[3]);
    spec.output_height = static_cast<int>(input_shape_[2]);
    spec.resize_mode =
        vision_operators::PreprocessResizeMode::kLetterbox;
    spec.output_rgb = true;
    spec.scale = {
        1.0F / 255.0F,
        1.0F / 255.0F,
        1.0F / 255.0F};
    spec.padding = {0.0F, 0.0F, 0.0F};
    auto prepared = prepare_image(
        input, spec,
        [this](const cv::Mat& bgr) {
            return preprocess(bgr);
        });
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    // Run inference using base class method
    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session(prepared.tensor());
    const auto t_infer1 = std::chrono::steady_clock::now();
    prepared.complete();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    // Postprocess
    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::PoseResultList results = postprocess_impl(
        outputs, orig_size, use_conf, use_iou, use_kp, max_det);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


std::vector<vision_core::InferIntent> YOLOv8PoseDetector::supported_intents() const {
    return {vision_core::InferIntent::kEstimatePose};
}

vision_core::InferResponse YOLOv8PoseDetector::RunImpl(const vision_core::InferRequest& request) {
    const auto& image_input = std::get<vision_core::ImageInput>(request.input);

    vision_common::PoseResultList task_results =
        estimate_pose_input(
            image_input,
            request.params.conf_threshold,
            request.params.iou_threshold, request.params.kp_threshold,
            request.params.max_det);
    return vision_core::make_infer_response(std::move(task_results));
}

std::vector<vision_core::ModelCapability> YOLOv8PoseDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::PoseResultList YOLOv8PoseDetector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold) {

    return postprocess_impl(outputs, orig_size, conf_threshold, iou_threshold,
        point_confidence_threshold_, -1);
}

vision_common::PoseResultList YOLOv8PoseDetector::postprocess_impl(
    std::vector<Ort::Value>& outputs, const cv::Size& orig_size,
    float conf_threshold, float iou_threshold, float kp_threshold, int max_det) {
    if (outputs.empty() || !outputs[0].IsTensor()) {
        throw std::runtime_error("YOLOv8-Pose expects a float32 [1,56,N] output");
    }
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 3 || shape[0] != 1 || shape[1] != 56 || shape[2] < 0 ||
        shape[2] > std::numeric_limits<int>::max() ||
        orig_size.width <= 0 || orig_size.height <= 0) {
        throw std::runtime_error("YOLOv8-Pose expects float32 [1,56,N] for 17 keypoints and a positive image size");
    }
    if (max_det == 0) return {};

    // Get output data
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    auto dets_tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dets_dims = dets_tensor_info.GetShape();

    int anchors = static_cast<int>(dets_dims[2]);  // 2100

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    float ratio = std::min(
        static_cast<float>(inputWidth) / static_cast<float>(orig_size.width),
        static_cast<float>(inputHeight) / static_cast<float>(orig_size.height));
    int unpad_w = static_cast<int>(std::round(orig_size.width * ratio));
    int unpad_h = static_cast<int>(std::round(orig_size.height * ratio));

    const float dw = std::round((inputWidth - unpad_w) / 2.0f - 0.1f);
    const float dh = std::round((inputHeight - unpad_h) / 2.0f - 0.1f);

    // First pass: extract all detections with confidence > threshold
    vision_common::PoseResultList objects;
    for (int j = 0; j < anchors; ++j) {
        if (std::isfinite(output_data[4 * anchors + j]) &&
            output_data[4 * anchors + j] > conf_threshold) {
            if (!std::isfinite(output_data[j]) || !std::isfinite(output_data[anchors + j]) ||
                !std::isfinite(output_data[2 * anchors + j]) ||
                !std::isfinite(output_data[3 * anchors + j]) ||
                output_data[2 * anchors + j] <= 0.0f || output_data[3 * anchors + j] <= 0.0f) continue;
            // Decode box
            float half_width = output_data[2 * anchors + j] / 2;
            float half_height = output_data[3 * anchors + j] / 2;
            float x1 = (output_data[j] - half_width - dw) / ratio;
            float y1 = (output_data[anchors + j] - half_height - dh) / ratio;
            float x2 = (output_data[j] + half_width - dw) / ratio;
            float y2 = (output_data[anchors + j] + half_height - dh) / ratio;

            if (!std::isfinite(x1) || !std::isfinite(y1) ||
                !std::isfinite(x2) || !std::isfinite(y2)) continue;
            // Clamp to image bounds
            const float max_x = static_cast<float>(std::max(orig_size.width - 1, 1));
            const float max_y = static_cast<float>(std::max(orig_size.height - 1, 1));
            x1 = std::clamp(x1, 0.0f, max_x);
            y1 = std::clamp(y1, 0.0f, max_y);
            x2 = std::clamp(x2, 0.0f, max_x);
            y2 = std::clamp(y2, 0.0f, max_y);
            if (x2 - x1 < 10.0f || y2 - y1 < 10.0f) {
                continue;
            }

            vision_common::PoseResult det;
            det.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
            det.score = output_data[4 * anchors + j];
            det.label = 0;  // Person class for pose detection

            // Decode keypoints and count visible ones
            det.keypoints.reserve(17);
            int visible_count = 0;
            bool finite_keypoints = true;
            for (int k = 0; k < 17; ++k) {
                vision_common::KeyPoint kp;
                kp.x = (output_data[(5 + k * 3) * anchors + j] - dw) / ratio;
                kp.y = (output_data[(5 + k * 3 + 1) * anchors + j] - dh) / ratio;
                kp.visibility = output_data[(5 + k * 3 + 2) * anchors + j];
                if (!std::isfinite(kp.x) || !std::isfinite(kp.y) || !std::isfinite(kp.visibility)) {
                    finite_keypoints = false;
                    break;
                }
                det.keypoints.push_back(kp);
                if (kp.visibility > kp_threshold) {
                    visible_count++;
                }
            }

            // Filter detections with too few visible keypoints (likely false positives)
            if (!finite_keypoints || visible_count < 3) {
                continue;
            }

            objects.push_back(std::move(det));
        }
    }

    // NMS preserves keypoints since they're already in the PoseResult objects
    auto results = nms(objects, iou_threshold);
    vision_common::limit_by_score(results, max_det,
        [](const auto& result) { return result.score; });
    return results;
}

float YOLOv8PoseDetector::calculate_iou(const vision_common::PoseResult& det1, const vision_common::PoseResult& det2) {
    // Use BoundingBox iou method
    return vision_common::iou(det1.bbox, det2.bbox);
}

vision_common::PoseResultList YOLOv8PoseDetector::nms(
    const vision_common::PoseResultList& dets,
    float iou_threshold) {
    if (dets.empty()) {
        return vision_common::PoseResultList();
    }

    // Convert Result to cv::Rect2f and scores for common nms function
    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    boxes.reserve(dets.size());
    scores.reserve(dets.size());

    for (const auto& det : dets) {
        boxes.push_back(cv::Rect2f(det.bbox.x1, det.bbox.y1, det.bbox.x2 - det.bbox.x1, det.bbox.y2 - det.bbox.y1));
        scores.push_back(det.score);
    }

    // Use common nms function
    std::vector<int> keep_indices = vision_common::nms(boxes, scores, iou_threshold);

    // Convert back to Result vector
    vision_common::PoseResultList final_dets;
    final_dets.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        final_dets.push_back(dets[idx]);
    }

    return final_dets;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv8PoseDetector> registrar("YOLOv8PoseDetector");

}  // namespace vision_deploy
