/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "av_tracker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "single_object_tracker_utils.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {

constexpr std::array<float, 3> kMean = {
    0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> kStd = {
    0.229f, 0.224f, 0.225f};

bool shape_is(
    const std::vector<int64_t>& actual,
    const std::array<int64_t, 4>& expected) {
    return actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin());
}

double elapsed_ms(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(
        end - begin).count();
}

}  // namespace

AVTracker::AVTracker(
    const std::string& model_path,
    int num_threads,
    bool lazy_load,
    std::string provider)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        provider_(std::move(provider)) {
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel> AVTracker::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for AVTracker");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<AVTracker>(
        model_path,
        vision_core::yaml_utils::getInt(params, "num_threads", 4),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void AVTracker::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (session_->GetInputCount() != 2 ||
        session_->GetOutputCount() != 3) {
        throw std::runtime_error(
            "AVTracker expects two inputs and three outputs");
    }
    const std::array<std::array<int64_t, 4>, 2> expected = {{
        {1, 3, kTemplateSize, kTemplateSize},
        {1, 3, kSearchSize, kSearchSize},
    }};
    for (size_t index = 0; index < expected.size(); ++index) {
        const Ort::TypeInfo type_info =
            session_->GetInputTypeInfo(index);
        const auto info =
            type_info.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            !shape_is(info.GetShape(), expected[index])) {
            throw std::runtime_error(
                "AVTracker input shapes must be "
                "[1,3,128,128] and [1,3,256,256]");
        }
    }
    model_loaded_ = true;
}

vision::Tracking AVTracker::initialize(
    const cv::Mat& image,
    const vision::BoundingBox& initial_box) {
    reset_runtime_profile();
    const auto total_begin = std::chrono::steady_clock::now();
    ensure_model_loaded();
    state_box_ =
        tracking_xyxy_to_xywh(initial_box, image.size());
    const auto preprocess_begin = std::chrono::steady_clock::now();
    template_tensor_ = preprocess_tracking_patch(
        image,
        state_box_,
        kTemplateFactor,
        kTemplateSize,
        kMean,
        kStd).values;
    initialized_ = true;
    const auto preprocess_end = std::chrono::steady_clock::now();
    const double preprocess_time =
        elapsed_ms(preprocess_begin, preprocess_end);
    const double total_time =
        elapsed_ms(total_begin, preprocess_end);
    set_runtime_preprocess_ms(preprocess_time);
    set_runtime_track_ms(total_time);
    set_runtime_total_ms(total_time);

    vision::Tracking output;
    output.bbox = initial_box;
    output.score = 1.0f;
    output.label = -1;
    output.track_id = 0;
    return output;
}

vision::Tracking AVTracker::track(const cv::Mat& image) {
    reset_runtime_profile();
    const auto total_begin = std::chrono::steady_clock::now();
    const auto preprocess_begin = std::chrono::steady_clock::now();
    TrackingTensor search = preprocess_tracking_patch(
        image,
        state_box_,
        kSearchFactor,
        kSearchSize,
        kMean,
        kStd);
    const auto preprocess_end = std::chrono::steady_clock::now();

    const std::array<int64_t, 4> template_shape = {
        1, 3, kTemplateSize, kTemplateSize};
    const std::array<int64_t, 4> search_shape = {
        1, 3, kSearchSize, kSearchSize};
    std::array<Ort::Value, 2> input_tensors = {
        Ort::Value::CreateTensor<float>(
            memory_info_,
            template_tensor_.data(),
            template_tensor_.size(),
            template_shape.data(),
            template_shape.size()),
        Ort::Value::CreateTensor<float>(
            memory_info_,
            search.values.data(),
            search.values.size(),
            search_shape.data(),
            search_shape.size()),
    };
    const auto infer_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_node_names_.data(),
        output_node_names_.size());
    const auto infer_end = std::chrono::steady_clock::now();

    const auto postprocess_begin = std::chrono::steady_clock::now();
    if (outputs.size() != 3) {
        throw std::runtime_error(
            "AVTracker returned an invalid output count");
    }
    for (const auto& output : outputs) {
        if (!output.IsTensor() ||
            output.GetTensorTypeAndShapeInfo().GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error(
                "AVTracker outputs must be float tensors");
        }
    }
    const auto score_shape =
        outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto size_shape =
        outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    const auto offset_shape =
        outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    if (score_shape.size() != 4 || size_shape.size() != 4 ||
        offset_shape.size() != 4 ||
        score_shape[0] != 1 || score_shape[1] != 1 ||
        score_shape[2] <= 0 || score_shape[3] <= 0 ||
        size_shape[0] != 1 || size_shape[1] != 2 ||
        size_shape[2] != score_shape[2] ||
        size_shape[3] != score_shape[3] ||
        offset_shape != size_shape) {
        throw std::runtime_error(
            "AVTracker expects score [1,1,H,W] and "
            "size/offset [1,2,H,W]");
    }
    const int feature_height =
        static_cast<int>(score_shape[2]);
    const int feature_width =
        static_cast<int>(score_shape[3]);
    const int feature_values =
        feature_height * feature_width;
    const float* score_map =
        outputs[0].GetTensorData<float>();
    const float* size_map =
        outputs[1].GetTensorData<float>();
    const float* offset_map =
        outputs[2].GetTensorData<float>();
    int maximum_index = 0;
    float maximum_score = score_map[0];
    for (int index = 1; index < feature_values; ++index) {
        if (score_map[index] > maximum_score) {
            maximum_index = index;
            maximum_score = score_map[index];
        }
    }
    if (!std::isfinite(maximum_score)) {
        throw std::runtime_error(
            "AVTracker returned a non-finite score");
    }
    const int feature_y = maximum_index / feature_width;
    const int feature_x = maximum_index % feature_width;
    const float size_width = size_map[maximum_index];
    const float size_height =
        size_map[feature_values + maximum_index];
    const float offset_x = offset_map[maximum_index];
    const float offset_y =
        offset_map[feature_values + maximum_index];
    if (!std::isfinite(size_width) ||
        !std::isfinite(size_height) ||
        !std::isfinite(offset_x) ||
        !std::isfinite(offset_y)) {
        throw std::runtime_error(
            "AVTracker returned non-finite box values");
    }

    const float search_side =
        static_cast<float>(kSearchSize) / search.resize_factor;
    const float center_x =
        (feature_x + offset_x) / feature_width * search_side;
    const float center_y =
        (feature_y + offset_y) / feature_height * search_side;
    const float predicted_width =
        std::max(kMinimumBoxSize, size_width * search_side);
    const float predicted_height =
        std::max(kMinimumBoxSize, size_height * search_side);
    const float previous_center_x =
        state_box_.x + 0.5f * state_box_.width;
    const float previous_center_y =
        state_box_.y + 0.5f * state_box_.height;
    const cv::Rect2f decoded(
        center_x + previous_center_x - 0.5f * search_side -
            0.5f * predicted_width,
        center_y + previous_center_y - 0.5f * search_side -
            0.5f * predicted_height,
        predicted_width,
        predicted_height);
    const vision::BoundingBox clipped =
        tracking_xywh_to_clipped_xyxy(
            decoded, image.size(), kMinimumBoxSize);
    state_box_ = cv::Rect2f(
        clipped.x1,
        clipped.y1,
        clipped.x2 - clipped.x1,
        clipped.y2 - clipped.y1);
    const auto postprocess_end = std::chrono::steady_clock::now();

    const double preprocess_time =
        elapsed_ms(preprocess_begin, preprocess_end);
    const double inference_time =
        elapsed_ms(infer_begin, infer_end);
    const double postprocess_time =
        elapsed_ms(postprocess_begin, postprocess_end);
    const double total_time =
        elapsed_ms(total_begin, postprocess_end);
    set_runtime_preprocess_ms(preprocess_time);
    set_runtime_model_infer_ms(inference_time);
    set_runtime_postprocess_ms(postprocess_time);
    set_runtime_track_ms(total_time);
    set_runtime_total_ms(total_time);
    add_runtime_component_timing("avtrack.infer", inference_time);

    vision::Tracking output;
    output.bbox = clipped;
    output.score = maximum_score;
    output.label = -1;
    output.track_id = 0;
    return output;
}

vision_core::InferResponse AVTracker::Run(
    const vision_core::InferRequest& request) {
    vision_core::InferResponse response;
    if (request.intent != vision_core::InferIntent::kTrack) {
        response.ok = false;
        response.error_message =
            "AVTracker only supports kTrack";
        return response;
    }
    const auto* input =
        std::get_if<vision_core::ImageInput>(&request.input);
    if (input == nullptr) {
        response.ok = false;
        response.error_message =
            "AVTracker expects ImageInput";
        return response;
    }
    if (!input->has_initial_bbox && !initialized_) {
        response.ok = false;
        response.error_message =
            "AVTracker requires an initial bounding box";
        return response;
    }
    const cv::Mat image =
        vision_operators::image_input_to_bgr_cpu(*input);
    response.results.emplace_back(
        input->has_initial_bbox
            ? initialize(image, input->initial_bbox)
            : track(image));
    return response;
}

std::vector<vision_core::InferIntent>
AVTracker::supported_intents() const {
    return {vision_core::InferIntent::kTrack};
}

std::vector<vision_core::ModelCapability>
AVTracker::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<AVTracker> registrar(
    "AVTracker");

}  // namespace vision_deploy
