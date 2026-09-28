/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rtmpose_estimator.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {
namespace {

constexpr int kInputWidth = 192;
constexpr int kInputHeight = 256;
constexpr int kKeypointCount = 17;
constexpr float kSimccSplitRatio = 2.0F;
constexpr std::array<float, 3> kMean = {123.675F, 116.28F, 103.53F};
constexpr std::array<float, 3> kStd = {58.395F, 57.12F, 57.375F};

cv::Mat preprocess(const cv::Mat& warped) {
    cv::Mat blob =
        cv::dnn::blobFromImage(warped, 1.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
    const int plane_size = kInputWidth * kInputHeight;
    float* data = blob.ptr<float>();
    for (int channel = 0; channel < 3; ++channel) {
        float* plane = data + channel * plane_size;
        for (int i = 0; i < plane_size; ++i) {
            plane[i] = (plane[i] - kMean[channel]) / kStd[channel];
        }
    }
    return blob;
}

std::pair<int, float> peak(const float* values, int count) {
    const float* best = std::max_element(values, values + count);
    return {static_cast<int>(best - values), *best};
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> RTMPoseEstimator::create(
    const YAML::Node& config, bool lazy_load) {
    const std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found for RTMPoseEstimator");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<RTMPoseEstimator>(
        model_path, vision_core::yaml_utils::getInt(params, "num_threads", 8),
        vision_core::yaml_utils::getProvider(config),
        vision_core::yaml_utils::getFloat(params, "conf_threshold", 0.25F), lazy_load);
}

RTMPoseEstimator::RTMPoseEstimator(const std::string& model_path, int num_threads,
    const std::string& provider, float confidence_threshold,
    bool lazy_load)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        provider_(provider),
        confidence_threshold_(confidence_threshold) {
    if (!lazy_load) load_model();
}

void RTMPoseEstimator::load_model() {
    if (model_loaded_) return;
    init_session(num_threads_, provider_);
    if (input_shape_ != std::vector<int64_t>{1, 3, kInputHeight, kInputWidth} ||
        output_names_.size() != 2) {
        throw std::runtime_error("RTMPose expects input [1,3,256,192] and two SimCC outputs");
    }
    const auto x_it = std::find(output_names_.begin(), output_names_.end(), "simcc_x");
    const auto y_it = std::find(output_names_.begin(), output_names_.end(), "simcc_y");
    if (x_it == output_names_.end() || y_it == output_names_.end()) {
        throw std::runtime_error("RTMPose expects simcc_x and simcc_y outputs");
    }
    simcc_x_index_ = static_cast<size_t>(x_it - output_names_.begin());
    simcc_y_index_ = static_cast<size_t>(y_it - output_names_.begin());
    model_loaded_ = true;
}

std::vector<vision_core::InferIntent> RTMPoseEstimator::supported_intents() const {
    return {vision_core::InferIntent::kEstimatePose};
}

std::vector<vision_core::ModelCapability> RTMPoseEstimator::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_core::InferResponse RTMPoseEstimator::Run(const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kEstimatePose) {
        return unsupported_intent_response(request.intent);
    }
    const auto* input = std::get_if<vision_core::ImageInput>(&request.input);
    if (input == nullptr || input->image.empty()) {
        return {{}, false, "RTMPose expects a non-empty ImageInput"};
    }
    cv::Mat bgr;
    if (input->format == vision_core::ImagePixelFormat::kNv12) {
        cv::cvtColor(input->image, bgr, cv::COLOR_YUV2BGR_NV12);
    } else {
        bgr = input->image;
    }
    if (bgr.type() != CV_8UC3) {
        return {{}, false, "RTMPose expects a BGR8 image (CV_8UC3)"};
    }
    ensure_model_loaded();
    reset_runtime_profile();
    const auto begin = std::chrono::steady_clock::now();
    float scale = 1.0F;
    float offset_x = 0.0F;
    float offset_y = 0.0F;
    cv::Mat warped;
    if (bgr.cols == kInputWidth && bgr.rows == kInputHeight) {
        warped = bgr;
    } else {
        scale = std::min(static_cast<float>(kInputWidth) / bgr.cols,
            static_cast<float>(kInputHeight) / bgr.rows);
        offset_x = (kInputWidth - bgr.cols * scale) * 0.5F;
        offset_y = (kInputHeight - bgr.rows * scale) * 0.5F;
        const cv::Matx23f affine(scale, 0.0F, offset_x, 0.0F, scale, offset_y);
        cv::warpAffine(bgr, warped, affine, cv::Size(kInputWidth, kInputHeight),
            cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    }
    const cv::Mat blob = preprocess(warped);
    const auto pre_end = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(
        std::chrono::duration<double, std::milli>(pre_end - begin).count());
    std::vector<Ort::Value> outputs = run_session(blob);
    const auto post_begin = std::chrono::steady_clock::now();
    const auto x_shape = outputs[simcc_x_index_].GetTensorTypeAndShapeInfo().GetShape();
    const auto y_shape = outputs[simcc_y_index_].GetTensorTypeAndShapeInfo().GetShape();
    if (x_shape != std::vector<int64_t>{1, kKeypointCount, kInputWidth * 2} ||
        y_shape != std::vector<int64_t>{1, kKeypointCount, kInputHeight * 2}) {
        throw std::runtime_error("RTMPose SimCC tensor shape mismatch");
    }
    const float* x_data = outputs[simcc_x_index_].GetTensorData<float>();
    const float* y_data = outputs[simcc_y_index_].GetTensorData<float>();
    vision_common::PoseResult pose;
    pose.bbox = {0.0F, 0.0F, static_cast<float>(bgr.cols), static_cast<float>(bgr.rows)};
    pose.label = 0;
    pose.keypoints.reserve(kKeypointCount);
    float score_sum = 0.0F;
    for (int i = 0; i < kKeypointCount; ++i) {
        const auto x_peak = peak(x_data + i * kInputWidth * 2, kInputWidth * 2);
        const auto y_peak = peak(y_data + i * kInputHeight * 2, kInputHeight * 2);
        // SimCC peak values are model scores, not probabilities in [0, 1].
        const float score = std::min(x_peak.second, y_peak.second);
        const float crop_x = x_peak.first / kSimccSplitRatio;
        const float crop_y = y_peak.first / kSimccSplitRatio;
        vision_common::KeyPoint point;
        point.x = (crop_x - offset_x) / scale;
        point.y = (crop_y - offset_y) / scale;
        point.visibility = score;
        pose.keypoints.push_back(point);
        score_sum += score;
    }
    pose.score = score_sum / kKeypointCount;
    vision_core::InferResponse response;
    const float threshold = request.params.conf_threshold > 0.0F
        ? request.params.conf_threshold
        : confidence_threshold_;
    if (pose.score >= threshold) response.results.emplace_back(std::move(pose));
    const auto end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(end - post_begin).count());
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(end - begin).count());
    return response;
}

static vision_core::ModelRegistrar<RTMPoseEstimator> registrar("RTMPoseEstimator");

}  // namespace vision_deploy
