/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mixformer_tracker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <sstream>
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

std::string shape_string(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << '[';
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << shape[index];
    }
    stream << ']';
    return stream.str();
}

double elapsed_ms(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(
        end - begin).count();
}

}  // namespace

MixFormerTracker::MixFormerTracker(
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

std::unique_ptr<vision_core::BaseModel> MixFormerTracker::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for MixFormerTracker");
    }
    const YAML::Node params = config["default_params"];
    auto tracker = std::make_unique<MixFormerTracker>(
        model_path,
        vision_core::yaml_utils::getInt(params, "num_threads", 4),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
    tracker->update_interval_ = std::max(
        1,
        vision_core::yaml_utils::getInt(
            params, "update_interval", 200));
    tracker->update_threshold_ =
        vision_core::yaml_utils::getFloat(
            params, "update_threshold", 0.5f);
    tracker->max_score_decay_ =
        vision_core::yaml_utils::getFloat(
            params, "max_score_decay", 1.0f);
    return tracker;
}

void MixFormerTracker::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (session_->GetInputCount() != 3 ||
        session_->GetOutputCount() != 2) {
        throw std::runtime_error(
            "MixFormerTracker expects three inputs and two outputs");
    }
    const std::array<std::array<int64_t, 4>, 3> expected = {{
        {1, 3, kTemplateSize, kTemplateSize},
        {1, 3, kTemplateSize, kTemplateSize},
        {1, 3, kSearchSize, kSearchSize},
    }};
    for (size_t index = 0; index < expected.size(); ++index) {
        const Ort::TypeInfo type_info =
            session_->GetInputTypeInfo(index);
        const auto info =
            type_info.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> actual_shape = info.GetShape();
        if (info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            !shape_is(actual_shape, expected[index])) {
            throw std::runtime_error(
                "MixFormerTracker input " + std::to_string(index) +
                " has shape " + shape_string(actual_shape) +
                "; expected [1,3,112,112], [1,3,112,112], "
                "[1,3,224,224]");
        }
    }
    model_loaded_ = true;
}

vision::Tracking MixFormerTracker::initialize(
    const cv::Mat& image,
    const vision::BoundingBox& initial_box) {
    reset_runtime_profile();
    const auto total_begin = std::chrono::steady_clock::now();
    ensure_model_loaded();
    state_box_ =
        tracking_xyxy_to_xywh(initial_box, image.size());
    const auto preprocess_begin = std::chrono::steady_clock::now();
    TrackingTensor patch = preprocess_tracking_patch(
        image,
        state_box_,
        kTemplateFactor,
        kTemplateSize,
        kMean,
        kStd);
    template_tensor_ = patch.values;
    online_template_tensor_ = patch.values;
    best_online_template_tensor_ = std::move(patch.values);
    frame_id_ = 0;
    max_score_ = 0.0f;
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

vision::Tracking MixFormerTracker::track(const cv::Mat& image) {
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
    std::array<Ort::Value, 3> input_tensors = {
        Ort::Value::CreateTensor<float>(
            memory_info_,
            template_tensor_.data(),
            template_tensor_.size(),
            template_shape.data(),
            template_shape.size()),
        Ort::Value::CreateTensor<float>(
            memory_info_,
            online_template_tensor_.data(),
            online_template_tensor_.size(),
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
    if (outputs.size() != 2 ||
        !outputs[0].IsTensor() || !outputs[1].IsTensor()) {
        throw std::runtime_error(
            "MixFormerTracker returned invalid outputs");
    }
    const auto box_info =
        outputs[0].GetTensorTypeAndShapeInfo();
    const auto score_info =
        outputs[1].GetTensorTypeAndShapeInfo();
    if (box_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        score_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        box_info.GetElementCount() != 4 ||
        score_info.GetElementCount() != 1) {
        throw std::runtime_error(
            "MixFormerTracker expects four box values and one score");
    }
    const float* prediction =
        outputs[0].GetTensorData<float>();
    const float score =
        outputs[1].GetTensorData<float>()[0];
    for (size_t index = 0; index < 4; ++index) {
        if (!std::isfinite(prediction[index])) {
            throw std::runtime_error(
                "MixFormerTracker returned a non-finite box");
        }
    }
    if (!std::isfinite(score)) {
        throw std::runtime_error(
            "MixFormerTracker returned a non-finite score");
    }

    const float search_side =
        static_cast<float>(kSearchSize) / search.resize_factor;
    const float predicted_width =
        std::max(kMinimumBoxSize, prediction[2] * search_side);
    const float predicted_height =
        std::max(kMinimumBoxSize, prediction[3] * search_side);
    const float previous_center_x =
        state_box_.x + 0.5f * state_box_.width;
    const float previous_center_y =
        state_box_.y + 0.5f * state_box_.height;
    const cv::Rect2f decoded(
        prediction[0] * search_side + previous_center_x -
            0.5f * search_side - 0.5f * predicted_width,
        prediction[1] * search_side + previous_center_y -
            0.5f * search_side - 0.5f * predicted_height,
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

    ++frame_id_;
    max_score_ *= max_score_decay_;
    if (score > max_score_ && score > update_threshold_) {
        // The online branch has the same [1,3,112,112] contract as the
        // fixed template branch. The independent demo incorrectly used the
        // 224 search size here, which creates a mismatched tensor.
        best_online_template_tensor_ = preprocess_tracking_patch(
            image,
            state_box_,
            kTemplateFactor,
            kTemplateSize,
            kMean,
            kStd).values;
        max_score_ = score;
    }
    if (frame_id_ >= update_interval_) {
        online_template_tensor_ = best_online_template_tensor_;
        frame_id_ = 0;
    }
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
    add_runtime_component_timing(
        "mixformer.infer", inference_time);

    vision::Tracking output;
    output.bbox = clipped;
    output.score = score;
    output.label = -1;
    output.track_id = 0;
    return output;
}

vision_core::InferResponse MixFormerTracker::Run(
    const vision_core::InferRequest& request) {
    vision_core::InferResponse response;
    if (request.intent != vision_core::InferIntent::kTrack) {
        response.ok = false;
        response.error_message =
            "MixFormerTracker only supports kTrack";
        return response;
    }
    const auto* input =
        std::get_if<vision_core::ImageInput>(&request.input);
    if (input == nullptr) {
        response.ok = false;
        response.error_message =
            "MixFormerTracker expects ImageInput";
        return response;
    }
    if (!input->has_initial_bbox && !initialized_) {
        response.ok = false;
        response.error_message =
            "MixFormerTracker requires an initial bounding box";
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
MixFormerTracker::supported_intents() const {
    return {vision_core::InferIntent::kTrack};
}

std::vector<vision_core::ModelCapability>
MixFormerTracker::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<MixFormerTracker> registrar(
    "MixFormerTracker");

}  // namespace vision_deploy
