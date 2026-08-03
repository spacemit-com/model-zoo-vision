/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nanotrack_tracker.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "single_object_tracker_utils.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {

double elapsed_ms(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(
        end - begin).count();
}

bool shape_is(
    const std::vector<int64_t>& actual,
    const std::array<int64_t, 4>& expected) {
    return actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin());
}

std::string shape_text(const std::vector<int64_t>& shape) {
    std::string output = "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            output += ",";
        }
        output += std::to_string(shape[index]);
    }
    output += "]";
    return output;
}

std::string resolve_auxiliary_model_path(const std::string& raw_path) {
    if (raw_path.empty()) {
        throw std::runtime_error(
            "NanoTracker auxiliary model path is empty");
    }
    std::string path = raw_path;
    if (path[0] == '~' &&
        (path.size() == 1 || path[1] == '/')) {
        const char* home = std::getenv("HOME");
        if (home != nullptr && home[0] != '\0') {
            path = std::string(home) + path.substr(1);
        }
    }
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(
            "NanoTracker auxiliary model file not found: " + path +
            ". Please run examples/nanotrack/scripts/download_models.sh");
    }
    return std::filesystem::path(path).lexically_normal().string();
}

void configure_session_options(
    Ort::SessionOptions* options,
    int num_threads,
    const std::string& provider,
    const char* role) {
    if (options == nullptr || num_threads <= 0) {
        throw std::invalid_argument(
            "NanoTracker session options are invalid");
    }
    options->SetIntraOpNumThreads(num_threads);
    options->SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider == "SpaceMITExecutionProvider") {
        Ort::Status status =
            Ort::SessionOptionsSpaceMITEnvInit(*options);
        if (!status.IsOK()) {
            throw std::runtime_error(
                std::string("SpaceMIT EP init failed (NanoTrack ") +
                role + "): " + status.GetErrorMessage());
        }
    }
}

void collect_node_names(
    Ort::Session* session,
    bool inputs,
    Ort::AllocatorWithDefaultOptions* allocator,
    std::vector<std::string>* storage,
    std::vector<const char*>* names) {
    if (session == nullptr || allocator == nullptr ||
        storage == nullptr || names == nullptr) {
        throw std::invalid_argument(
            "NanoTracker node-name arguments are invalid");
    }
    const size_t count =
        inputs ? session->GetInputCount() : session->GetOutputCount();
    storage->resize(count);
    names->resize(count);
    for (size_t index = 0; index < count; ++index) {
        auto value = inputs
            ? session->GetInputNameAllocated(index, *allocator)
            : session->GetOutputNameAllocated(index, *allocator);
        (*storage)[index] = value.get();
    }
    for (size_t index = 0; index < count; ++index) {
        (*names)[index] = (*storage)[index].c_str();
    }
}

std::vector<float> make_hanning_window(int size) {
    if (size <= 1) {
        throw std::invalid_argument(
            "NanoTrack Hanning window size must be greater than one");
    }
    std::vector<float> one_dimensional(size);
    constexpr float kPi = 3.14159265358979323846f;
    for (int index = 0; index < size; ++index) {
        one_dimensional[index] =
            0.5f * (1.0f - std::cos(
                2.0f * kPi * index / (size - 1)));
    }
    std::vector<float> output(
        static_cast<size_t>(size) * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            output[static_cast<size_t>(y) * size + x] =
                one_dimensional[y] * one_dimensional[x];
        }
    }
    return output;
}

float stable_change(float ratio) {
    if (!std::isfinite(ratio) || ratio <= 0.0f) {
        throw std::runtime_error(
            "NanoTracker decoded an invalid size ratio");
    }
    return std::max(ratio, 1.0f / ratio);
}

float padded_size(float width, float height) {
    const float padding = 0.5f * (width + height);
    return std::sqrt(
        (width + padding) * (height + padding));
}

void validate_params(const NanoTrackParams& params) {
    if (params.num_threads <= 0 ||
        params.template_num_threads <= 0 ||
        params.head_num_threads <= 0 ||
        !std::isfinite(params.context_amount) ||
        params.context_amount < 0.0f ||
        !std::isfinite(params.penalty_k) ||
        params.penalty_k < 0.0f ||
        !std::isfinite(params.window_influence) ||
        params.window_influence < 0.0f ||
        params.window_influence > 1.0f ||
        !std::isfinite(params.learning_rate) ||
        params.learning_rate < 0.0f ||
        params.learning_rate > 1.0f) {
        throw std::invalid_argument(
            "NanoTracker parameters are invalid");
    }
}

}  // namespace

std::vector<cv::Point2f> generate_nanotrack_points(
    int stride,
    int score_size) {
    if (stride <= 0 || score_size <= 0) {
        throw std::invalid_argument(
            "NanoTrack point stride and score size must be positive");
    }
    const float origin =
        -0.5f * static_cast<float>(score_size * stride);
    std::vector<cv::Point2f> points;
    points.reserve(
        static_cast<size_t>(score_size) * score_size);
    for (int y = 0; y < score_size; ++y) {
        for (int x = 0; x < score_size; ++x) {
            points.emplace_back(
                origin + x * stride,
                origin + y * stride);
        }
    }
    return points;
}

std::vector<float> nanotrack_foreground_probabilities(
    const float* logits,
    int score_size) {
    if (logits == nullptr || score_size <= 0) {
        throw std::invalid_argument(
            "NanoTrack classification logits are invalid");
    }
    const int values = score_size * score_size;
    std::vector<float> probabilities(values);
    for (int index = 0; index < values; ++index) {
        const float background = logits[index];
        const float foreground = logits[values + index];
        if (!std::isfinite(background) ||
            !std::isfinite(foreground)) {
            throw std::runtime_error(
                "NanoTracker returned non-finite classification logits");
        }
        const float maximum = std::max(background, foreground);
        const float background_exp =
            std::exp(background - maximum);
        const float foreground_exp =
            std::exp(foreground - maximum);
        probabilities[index] =
            foreground_exp / (background_exp + foreground_exp);
    }
    return probabilities;
}

NanoTracker::NanoTracker(
    const std::string& search_model_path,
    std::string template_model_path,
    std::string head_model_path,
    NanoTrackParams params,
    bool lazy_load,
    std::string provider)
    : BaseModel(search_model_path, lazy_load),
        template_model_path_(std::move(template_model_path)),
        head_model_path_(std::move(head_model_path)),
        params_(params),
        provider_(std::move(provider)),
        points_(generate_nanotrack_points(
            kPointStride, kScoreSize)),
        window_(make_hanning_window(kScoreSize)) {
    validate_params(params_);
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel> NanoTracker::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string search_model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    const YAML::Node params = config["default_params"];
    const std::string template_model_path =
        vision_core::yaml_utils::getString(
            params, "template_model_path");
    const std::string head_model_path =
        vision_core::yaml_utils::getString(
            params, "head_model_path");
    if (search_model_path.empty() ||
        template_model_path.empty() ||
        head_model_path.empty()) {
        throw std::runtime_error(
            "NanoTracker requires model_path, "
            "default_params.template_model_path, and "
            "default_params.head_model_path");
    }

    NanoTrackParams values;
    values.num_threads =
        vision_core::yaml_utils::getInt(
            params, "num_threads", values.num_threads);
    values.template_num_threads =
        vision_core::yaml_utils::getInt(
            params,
            "template_num_threads",
            values.template_num_threads);
    values.head_num_threads =
        vision_core::yaml_utils::getInt(
            params, "head_num_threads", values.head_num_threads);
    values.context_amount =
        vision_core::yaml_utils::getFloat(
            params, "context_amount", values.context_amount);
    values.penalty_k =
        vision_core::yaml_utils::getFloat(
            params, "penalty_k", values.penalty_k);
    values.window_influence =
        vision_core::yaml_utils::getFloat(
            params,
            "window_influence",
            values.window_influence);
    values.learning_rate =
        vision_core::yaml_utils::getFloat(
            params, "learning_rate", values.learning_rate);

    return std::make_unique<NanoTracker>(
        search_model_path,
        template_model_path,
        head_model_path,
        values,
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void NanoTracker::initialize_auxiliary_sessions() {
    template_model_path_ =
        resolve_auxiliary_model_path(template_model_path_);
    head_model_path_ =
        resolve_auxiliary_model_path(head_model_path_);

    Ort::SessionOptions template_options;
    configure_session_options(
        &template_options,
        params_.template_num_threads,
        "CPUExecutionProvider",
        "template backbone");
    template_session_ = std::make_unique<Ort::Session>(
        vision_core::shared_ort_env(),
        template_model_path_.c_str(),
        template_options);

    Ort::SessionOptions head_options;
    configure_session_options(
        &head_options,
        params_.head_num_threads,
        provider_,
        "head");
    head_session_ = std::make_unique<Ort::Session>(
        vision_core::shared_ort_env(),
        head_model_path_.c_str(),
        head_options);

    collect_node_names(
        template_session_.get(),
        true,
        &allocator_,
        &template_input_names_storage_,
        &template_input_names_);
    collect_node_names(
        template_session_.get(),
        false,
        &allocator_,
        &template_output_names_storage_,
        &template_output_names_);
    collect_node_names(
        head_session_.get(),
        true,
        &allocator_,
        &head_input_names_storage_,
        &head_input_names_);
    collect_node_names(
        head_session_.get(),
        false,
        &allocator_,
        &head_output_names_storage_,
        &head_output_names_);
}

void NanoTracker::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(params_.num_threads, provider_);
    initialize_auxiliary_sessions();

    if (session_->GetInputCount() != 1 ||
        session_->GetOutputCount() != 1 ||
        !shape_is(
            input_shape_,
            {1, 3, kSearchSize, kSearchSize})) {
        throw std::runtime_error(
            "NanoTracker search backbone expects "
            "[1,3,255,255] input and one output");
    }
    const Ort::TypeInfo search_input_type =
        session_->GetInputTypeInfo(0);
    const Ort::TypeInfo search_output_type =
        session_->GetOutputTypeInfo(0);
    const auto search_input_info =
        search_input_type.GetTensorTypeAndShapeInfo();
    const auto search_output_info =
        search_output_type.GetTensorTypeAndShapeInfo();
    if (search_input_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        search_output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        !shape_is(
            search_output_info.GetShape(),
            {1, 96, kScoreSize, kScoreSize})) {
        throw std::runtime_error(
            "NanoTracker search backbone contract must be "
            "[1,3,255,255] -> [1,96,16,16] float32, got input type " +
            std::to_string(search_input_info.GetElementType()) +
            " and output " +
            shape_text(search_output_info.GetShape()) +
            " type " +
            std::to_string(search_output_info.GetElementType()));
    }

    if (template_session_->GetInputCount() != 1 ||
        template_session_->GetOutputCount() != 1) {
        throw std::runtime_error(
            "NanoTracker template backbone expects one input and one output");
    }
    const Ort::TypeInfo template_input_type =
        template_session_->GetInputTypeInfo(0);
    const Ort::TypeInfo template_output_type =
        template_session_->GetOutputTypeInfo(0);
    const auto template_input_info =
        template_input_type.GetTensorTypeAndShapeInfo();
    const auto template_output_info =
        template_output_type.GetTensorTypeAndShapeInfo();
    if (template_input_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        !shape_is(
            template_input_info.GetShape(),
            {1, 3, kTemplateSize, kTemplateSize}) ||
        template_output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        !shape_is(
            template_output_info.GetShape(),
            {1, 96, 8, 8})) {
        throw std::runtime_error(
            "NanoTracker template backbone contract must be "
            "[1,3,127,127] -> [1,96,8,8] float32");
    }

    if (head_session_->GetInputCount() != 2 ||
        head_session_->GetOutputCount() != 2) {
        throw std::runtime_error(
            "NanoTracker head expects two inputs and two outputs");
    }
    const std::array<std::array<int64_t, 4>, 2> head_inputs = {{
        {1, 96, 8, 8},
        {1, 96, kScoreSize, kScoreSize},
    }};
    const std::array<std::array<int64_t, 4>, 2> head_outputs = {{
        {1, 2, kScoreSize, kScoreSize},
        {1, 4, kScoreSize, kScoreSize},
    }};
    for (size_t index = 0; index < head_inputs.size(); ++index) {
        const Ort::TypeInfo input_type =
            head_session_->GetInputTypeInfo(index);
        const Ort::TypeInfo output_type =
            head_session_->GetOutputTypeInfo(index);
        const auto input_info =
            input_type.GetTensorTypeAndShapeInfo();
        const auto output_info =
            output_type.GetTensorTypeAndShapeInfo();
        if (input_info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            !shape_is(input_info.GetShape(), head_inputs[index]) ||
            output_info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            !shape_is(output_info.GetShape(), head_outputs[index])) {
            throw std::runtime_error(
                "NanoTracker head input/output shapes are incompatible");
        }
    }
    model_loaded_ = true;
}

void NanoTracker::release() {
    template_session_.reset();
    head_session_.reset();
    template_input_names_storage_.clear();
    template_output_names_storage_.clear();
    template_input_names_.clear();
    template_output_names_.clear();
    head_input_names_storage_.clear();
    head_output_names_storage_.clear();
    head_input_names_.clear();
    head_output_names_.clear();
    template_features_.clear();
    template_feature_shape_.clear();
    initialized_ = false;
    BaseModel::release();
}

cv::Mat NanoTracker::preprocess_crop(
    const cv::Mat& image,
    const cv::Point2f& center,
    int output_size,
    int crop_size) const {
    if (image.empty() || image.type() != CV_8UC3 ||
        output_size <= 0 || crop_size <= 0 ||
        !std::isfinite(center.x) ||
        !std::isfinite(center.y)) {
        throw std::invalid_argument(
            "NanoTracker crop arguments are invalid");
    }
    crop_size = std::min(
        crop_size,
        std::max(image.rows, image.cols));
    const float context = 0.5f * (crop_size + 1);
    const int x1 = static_cast<int>(
        std::floor(center.x - context + 0.5f));
    const int y1 = static_cast<int>(
        std::floor(center.y - context + 0.5f));
    const int x2 = x1 + crop_size - 1;
    const int y2 = y1 + crop_size - 1;
    const int left = std::max(0, -x1);
    const int top = std::max(0, -y1);
    const int right = std::max(0, x2 - image.cols + 1);
    const int bottom = std::max(0, y2 - image.rows + 1);

    cv::Mat padded;
    cv::copyMakeBorder(
        image,
        padded,
        top,
        bottom,
        left,
        right,
        cv::BORDER_CONSTANT,
        channel_average_);
    const cv::Rect roi(
        x1 + left,
        y1 + top,
        crop_size,
        crop_size);
    cv::Mat patch = padded(roi).clone();
    if (crop_size != output_size) {
        cv::resize(
            patch,
            patch,
            cv::Size(output_size, output_size),
            0.0,
            0.0,
            cv::INTER_LINEAR);
    }
    return cv::dnn::blobFromImage(
        patch,
        1.0,
        cv::Size(output_size, output_size),
        cv::Scalar(),
        false,
        false,
        CV_32F);
}

vision::Tracking NanoTracker::initialize(
    const cv::Mat& image,
    const vision::BoundingBox& initial_box) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto total_begin = std::chrono::steady_clock::now();
    state_box_ =
        tracking_xyxy_to_xywh(initial_box, image.size());
    channel_average_ = cv::mean(image);
    state_center_ = cv::Point2f(
        state_box_.x + 0.5f * (state_box_.width - 1.0f),
        state_box_.y + 0.5f * (state_box_.height - 1.0f));
    const float context =
        params_.context_amount *
        (state_box_.width + state_box_.height);
    const float template_side = std::sqrt(
        (state_box_.width + context) *
        (state_box_.height + context));

    const auto preprocess_begin = std::chrono::steady_clock::now();
    cv::Mat template_blob = preprocess_crop(
        image,
        state_center_,
        kTemplateSize,
        static_cast<int>(std::round(template_side)));
    const auto preprocess_end = std::chrono::steady_clock::now();

    const std::array<int64_t, 4> shape = {
        1, 3, kTemplateSize, kTemplateSize};
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info_,
        template_blob.ptr<float>(),
        template_blob.total(),
        shape.data(),
        shape.size());
    const auto infer_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = template_session_->Run(
        Ort::RunOptions{nullptr},
        template_input_names_.data(),
        &input,
        1,
        template_output_names_.data(),
        template_output_names_.size());
    const auto infer_end = std::chrono::steady_clock::now();
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error(
            "NanoTracker template backbone returned invalid output");
    }
    const auto info =
        outputs[0].GetTensorTypeAndShapeInfo();
    template_feature_shape_ = info.GetShape();
    const size_t values = info.GetElementCount();
    const float* data = outputs[0].GetTensorData<float>();
    template_features_.assign(data, data + values);
    initialized_ = true;

    const auto postprocess_end = std::chrono::steady_clock::now();
    const double preprocess_time =
        elapsed_ms(preprocess_begin, preprocess_end);
    const double inference_time =
        elapsed_ms(infer_begin, infer_end);
    const double postprocess_time =
        elapsed_ms(infer_end, postprocess_end);
    const double total_time =
        elapsed_ms(total_begin, postprocess_end);
    set_runtime_preprocess_ms(preprocess_time);
    set_runtime_model_infer_ms(inference_time);
    set_runtime_postprocess_ms(postprocess_time);
    set_runtime_track_ms(total_time);
    set_runtime_total_ms(total_time);
    add_runtime_component_timing(
        "nanotrack.template_backbone.infer",
        inference_time);

    vision::Tracking output;
    output.bbox = initial_box;
    output.score = 1.0f;
    output.label = -1;
    output.track_id = 0;
    return output;
}

vision::Tracking NanoTracker::track(const cv::Mat& image) {
    reset_runtime_profile();
    const auto total_begin = std::chrono::steady_clock::now();
    const float context =
        params_.context_amount *
        (state_box_.width + state_box_.height);
    const float template_side = std::sqrt(
        (state_box_.width + context) *
        (state_box_.height + context));
    const float scale_z =
        static_cast<float>(kTemplateSize) / template_side;
    const float search_side =
        template_side *
        static_cast<float>(kSearchSize) / kTemplateSize;
    const auto preprocess_begin = std::chrono::steady_clock::now();
    cv::Mat search_blob = preprocess_crop(
        image,
        state_center_,
        kSearchSize,
        static_cast<int>(std::round(search_side)));
    const auto preprocess_end = std::chrono::steady_clock::now();

    const std::array<int64_t, 4> search_shape = {
        1, 3, kSearchSize, kSearchSize};
    Ort::Value search_input = Ort::Value::CreateTensor<float>(
        memory_info_,
        search_blob.ptr<float>(),
        search_blob.total(),
        search_shape.data(),
        search_shape.size());
    const auto search_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value> search_outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &search_input,
        1,
        output_node_names_.data(),
        output_node_names_.size());
    const auto search_end = std::chrono::steady_clock::now();
    if (search_outputs.size() != 1 ||
        !search_outputs[0].IsTensor()) {
        throw std::runtime_error(
            "NanoTracker search backbone returned invalid output");
    }

    Ort::Value template_input = Ort::Value::CreateTensor<float>(
        memory_info_,
        template_features_.data(),
        template_features_.size(),
        template_feature_shape_.data(),
        template_feature_shape_.size());
    std::array<Ort::Value, 2> head_inputs = {
        std::move(template_input),
        std::move(search_outputs[0]),
    };
    const auto head_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = head_session_->Run(
        Ort::RunOptions{nullptr},
        head_input_names_.data(),
        head_inputs.data(),
        head_inputs.size(),
        head_output_names_.data(),
        head_output_names_.size());
    const auto head_end = std::chrono::steady_clock::now();
    if (outputs.size() != 2 ||
        !outputs[0].IsTensor() ||
        !outputs[1].IsTensor()) {
        throw std::runtime_error(
            "NanoTracker head returned invalid outputs");
    }

    const auto postprocess_begin = std::chrono::steady_clock::now();
    const auto classification_info =
        outputs[0].GetTensorTypeAndShapeInfo();
    const auto regression_info =
        outputs[1].GetTensorTypeAndShapeInfo();
    if (classification_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        regression_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        !shape_is(
            classification_info.GetShape(),
            {1, 2, kScoreSize, kScoreSize}) ||
        !shape_is(
            regression_info.GetShape(),
            {1, 4, kScoreSize, kScoreSize})) {
        throw std::runtime_error(
            "NanoTracker head output shapes are incompatible");
    }

    const std::vector<float> scores =
        nanotrack_foreground_probabilities(
            outputs[0].GetTensorData<float>(),
            kScoreSize);
    const float* regression =
        outputs[1].GetTensorData<float>();
    const int values = kScoreSize * kScoreSize;
    std::vector<float> centers_x(values);
    std::vector<float> centers_y(values);
    std::vector<float> widths(values);
    std::vector<float> heights(values);
    std::vector<float> penalties(values);
    std::vector<float> penalized_scores(values);
    for (int index = 0; index < values; ++index) {
        const float left =
            points_[index].x - regression[index];
        const float top =
            points_[index].y - regression[values + index];
        const float right =
            points_[index].x + regression[2 * values + index];
        const float bottom =
            points_[index].y + regression[3 * values + index];
        centers_x[index] = 0.5f * (left + right);
        centers_y[index] = 0.5f * (top + bottom);
        widths[index] = right - left;
        heights[index] = bottom - top;
        if (!std::isfinite(centers_x[index]) ||
            !std::isfinite(centers_y[index]) ||
            !std::isfinite(widths[index]) ||
            !std::isfinite(heights[index]) ||
            widths[index] <= 0.0f ||
            heights[index] <= 0.0f) {
            throw std::runtime_error(
                "NanoTracker returned invalid box regression values");
        }
        const float size_penalty = stable_change(
            padded_size(widths[index], heights[index]) /
            padded_size(
                state_box_.width * scale_z,
                state_box_.height * scale_z));
        const float ratio_penalty = stable_change(
            (state_box_.width / state_box_.height) /
            (widths[index] / heights[index]));
        penalties[index] = std::exp(
            -(ratio_penalty * size_penalty - 1.0f) *
            params_.penalty_k);
        penalized_scores[index] =
            penalties[index] * scores[index] *
                (1.0f - params_.window_influence) +
            window_[index] * params_.window_influence;
    }

    const int best_index = static_cast<int>(
        std::distance(
            penalized_scores.begin(),
            std::max_element(
                penalized_scores.begin(),
                penalized_scores.end())));
    const float learning_rate =
        penalties[best_index] *
        scores[best_index] *
        params_.learning_rate;
    const float predicted_center_x =
        state_center_.x + centers_x[best_index] / scale_z;
    const float predicted_center_y =
        state_center_.y + centers_y[best_index] / scale_z;
    const float predicted_width =
        state_box_.width * (1.0f - learning_rate) +
        widths[best_index] / scale_z * learning_rate;
    const float predicted_height =
        state_box_.height * (1.0f - learning_rate) +
        heights[best_index] / scale_z * learning_rate;
    const cv::Rect2f decoded(
        predicted_center_x - 0.5f * predicted_width,
        predicted_center_y - 0.5f * predicted_height,
        predicted_width,
        predicted_height);
    const vision::BoundingBox clipped =
        tracking_xywh_to_clipped_xyxy(
            decoded,
            image.size(),
            kMinimumBoxSize);
    state_box_ = cv::Rect2f(
        clipped.x1,
        clipped.y1,
        clipped.x2 - clipped.x1,
        clipped.y2 - clipped.y1);
    state_center_ = cv::Point2f(
        state_box_.x + 0.5f * state_box_.width,
        state_box_.y + 0.5f * state_box_.height);
    const auto postprocess_end = std::chrono::steady_clock::now();

    const double search_time =
        elapsed_ms(search_begin, search_end);
    const double head_time =
        elapsed_ms(head_begin, head_end);
    const double preprocess_time =
        elapsed_ms(preprocess_begin, preprocess_end);
    const double postprocess_time =
        elapsed_ms(postprocess_begin, postprocess_end);
    const double total_time =
        elapsed_ms(total_begin, postprocess_end);
    set_runtime_preprocess_ms(preprocess_time);
    set_runtime_model_infer_ms(search_time + head_time);
    set_runtime_postprocess_ms(postprocess_time);
    set_runtime_track_ms(total_time);
    set_runtime_total_ms(total_time);
    add_runtime_component_timing(
        "nanotrack.search_backbone.infer",
        search_time);
    add_runtime_component_timing(
        "nanotrack.head.infer",
        head_time);

    vision::Tracking output;
    output.bbox = clipped;
    output.score = scores[best_index];
    output.label = -1;
    output.track_id = 0;
    return output;
}

vision_core::InferResponse NanoTracker::Run(
    const vision_core::InferRequest& request) {
    vision_core::InferResponse response;
    if (request.intent != vision_core::InferIntent::kTrack) {
        response.ok = false;
        response.error_message =
            "NanoTracker only supports kTrack";
        return response;
    }
    const auto* input =
        std::get_if<vision_core::ImageInput>(&request.input);
    if (input == nullptr) {
        response.ok = false;
        response.error_message =
            "NanoTracker expects ImageInput";
        return response;
    }
    if (!input->has_initial_bbox && !initialized_) {
        response.ok = false;
        response.error_message =
            "NanoTracker requires an initial bounding box";
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
NanoTracker::supported_intents() const {
    return {vision_core::InferIntent::kTrack};
}

std::vector<vision_core::ModelCapability>
NanoTracker::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<NanoTracker> registrar(
    "NanoTracker");

}  // namespace vision_deploy
