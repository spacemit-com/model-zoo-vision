/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "landmark2d106.h"

#include <algorithm>
#include <chrono>
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
#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> Landmark2d106::create(const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(config, {"input_mean", "input_std"});
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for Landmark2d106");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for Landmark2d106");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    float input_mean = vision_core::yaml_utils::getFloat(default_params, "input_mean", 127.5f);
    float input_std = vision_core::yaml_utils::getFloat(default_params, "input_std", 128.0f);
    return std::make_unique<Landmark2d106>(
        model_path, num_threads, lazy_load, provider, input_mean, input_std);
}

Landmark2d106::Landmark2d106(const std::string& model_path,
                            int num_threads,
                            bool lazy_load,
                            const std::string& provider,
                            float input_mean,
                            float input_std)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        provider_(provider),
        input_mean_(input_mean),
        input_std_(input_std) {
    if (!std::isfinite(input_mean_) || !std::isfinite(input_std_) || input_std_ <= 0.0f ||
        !std::isfinite(1.0f / input_std_)) {
        throw std::invalid_argument("Landmark2d106 normalization must be finite with positive std and finite reciprocal");
    }
    if (!lazy_load) {
        load_model();
    }
}

void Landmark2d106::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_shape_.size() != 4 || input_shape_[0] > 1 || input_shape_[1] != 3 ||
        input_shape_[2] <= 0 || input_shape_[2] != input_shape_[3] ||
        input_shape_[2] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Landmark2d106 expects input [1,3,S,S] with fixed positive S");
    }
    input_size_ = static_cast<int>(input_shape_[2]);
    model_loaded_ = true;
}

cv::Mat Landmark2d106::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();

    const float inv_std = static_cast<float>(1.0 / input_std_);
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = input_size_;
    spec.output_height = input_size_;
    spec.mean = {input_mean_, input_mean_, input_mean_};
    spec.scale = {inv_std, inv_std, inv_std};
    return vision_operators::preprocess_bgr_to_nchw(image, spec);
}

vision_common::PoseResultList Landmark2d106::estimate_pose_input(
    const vision_core::ImageInput& input) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const cv::Size face_size(
        input.image.cols,
        input.format == vision_core::ImagePixelFormat::kNv12
            ? input.image.rows * 2 / 3
            : input.image.rows);
    const auto t_pre0 = std::chrono::steady_clock::now();
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = input_size_;
    spec.output_height = input_size_;
    spec.output_rgb = true;
    spec.mean = {input_mean_, input_mean_, input_mean_};
    spec.scale = {
        1.0F / input_std_,
        1.0F / input_std_,
        1.0F / input_std_};
    auto prepared = prepare_image(
        input, spec,
        [this](const cv::Mat& bgr) {
            return preprocess(bgr);
        });
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session(prepared.tensor());
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::PoseResultList results = postprocess(outputs, face_size);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

std::vector<vision_core::InferIntent> Landmark2d106::supported_intents() const {
    return {vision_core::InferIntent::kEstimatePose};
}

vision_core::InferResponse Landmark2d106::RunImpl(const vision_core::InferRequest& request) {
    const auto& image_input = std::get<vision_core::ImageInput>(request.input);

    vision_common::PoseResultList task_results =
        estimate_pose_input(image_input);
    vision_core::InferResponse response;
    if (request.params.max_det == 0) return response;
    return vision_core::make_infer_response(std::move(task_results));
}

std::vector<vision_core::ModelCapability> Landmark2d106::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::PoseResultList Landmark2d106::postprocess(std::vector<Ort::Value>& outputs,
                                                        const cv::Size& face_size) {
    if (outputs.empty() || !outputs[0].IsTensor()) {
        throw std::runtime_error("Landmark2d106 expects a float32 tensor of 106 coordinate pairs");
    }
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        (shape != std::vector<int64_t>{212} && shape != std::vector<int64_t>{1, 212} &&
        shape != std::vector<int64_t>{1, 106, 2}) || face_size.width <= 0 || face_size.height <= 0) {
        throw std::runtime_error("Landmark2d106 expects float32 [212], [1,212] or [1,106,2] output");
    }
    const float* landmarks_data = outputs[0].GetTensorData<float>();
    if (!std::all_of(landmarks_data, landmarks_data + 212,
        [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("Landmark2d106 returned non-finite coordinates");
    }
    constexpr int num_points = 106;

    vision_common::PoseResult det;
    det.bbox = vision_common::BoundingBox{0.0f, 0.0f,
                                        static_cast<float>(face_size.width),
                                        static_cast<float>(face_size.height)};
    det.score = 1.0f;
    det.label = 0;
    det.keypoints.reserve(static_cast<size_t>(num_points));
    for (int i = 0; i < num_points; ++i) {
        vision_common::KeyPoint kp;
        const float x_norm = landmarks_data[i * 2];
        const float y_norm = landmarks_data[i * 2 + 1];
        const float x_input = (x_norm + 1.0f) * (static_cast<float>(input_size_) * 0.5f);
        const float y_input = (y_norm + 1.0f) * (static_cast<float>(input_size_) * 0.5f);
        kp.x = x_input * static_cast<float>(face_size.width) / static_cast<float>(input_size_);
        kp.y = y_input * static_cast<float>(face_size.height) / static_cast<float>(input_size_);
        if (!std::isfinite(kp.x) || !std::isfinite(kp.y)) {
            throw std::runtime_error("Landmark2d106 coordinate transform overflowed");
        }
        kp.visibility = 1.0f;
        det.keypoints.push_back(kp);
    }
    return {det};
}

static vision_core::ModelRegistrar<Landmark2d106> registrar("Landmark2d106");

}  // namespace vision_deploy
