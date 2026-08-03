/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lightglue_matcher.h"

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

#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::string validate_lightglue_features(
    const vision::LocalFeatures& features,
    const std::string& expected_feature_type,
    int expected_keypoints,
    int expected_descriptor_dim) {
    if (features.feature_type != expected_feature_type) {
        return "feature_type '" + features.feature_type +
            "' is incompatible with expected '" +
            expected_feature_type + "'";
    }
    if (features.image_width <= 0 || features.image_height <= 0) {
        return "feature image dimensions must be positive";
    }
    if (features.descriptor_dim != expected_descriptor_dim) {
        return "descriptor_dim is incompatible with LightGlue weights";
    }
    if (static_cast<int>(features.keypoints.size()) !=
        expected_keypoints) {
        return "keypoint count is incompatible with LightGlue weights";
    }
    const size_t expected_values =
        static_cast<size_t>(expected_keypoints) *
        expected_descriptor_dim;
    if (features.descriptors.size() != expected_values) {
        return "descriptor buffer length does not match keypoints x descriptor_dim";
    }
    for (const auto& point : features.keypoints) {
        if (!std::isfinite(point.x) ||
            !std::isfinite(point.y) ||
            !std::isfinite(point.visibility) ||
            point.x < 0.0f ||
            point.y < 0.0f ||
            point.x >= static_cast<float>(features.image_width) ||
            point.y >= static_cast<float>(features.image_height)) {
            return "keypoint coordinates or score are invalid";
        }
    }
    for (const float value : features.descriptors) {
        if (!std::isfinite(value)) {
            return "descriptors must contain only finite values";
        }
    }
    return {};
}

std::vector<vision::FeatureMatch> filter_lightglue_matches(
    const float* log_scores,
    int keypoint_count,
    const vision::LocalFeatures& query,
    const vision::LocalFeatures& train,
    float filter_threshold) {
    if (log_scores == nullptr || keypoint_count <= 0 ||
        static_cast<int>(query.keypoints.size()) < keypoint_count ||
        static_cast<int>(train.keypoints.size()) < keypoint_count) {
        throw std::invalid_argument(
            "LightGlue match filtering input is invalid");
    }
    std::vector<int> query_to_train(keypoint_count, 0);
    std::vector<int> train_to_query(keypoint_count, 0);
    std::vector<float> query_scores(keypoint_count, 0.0f);
    for (int query_index = 0;
        query_index < keypoint_count;
        ++query_index) {
        int best = 0;
        float best_value =
            log_scores[query_index * keypoint_count];
        for (int train_index = 1;
            train_index < keypoint_count;
            ++train_index) {
            const float value =
                log_scores[
                    query_index * keypoint_count + train_index];
            if (value > best_value) {
                best_value = value;
                best = train_index;
            }
        }
        query_to_train[query_index] = best;
        query_scores[query_index] = best_value;
    }
    for (int train_index = 0;
        train_index < keypoint_count;
        ++train_index) {
        int best = 0;
        float best_value = log_scores[train_index];
        for (int query_index = 1;
            query_index < keypoint_count;
            ++query_index) {
            const float value =
                log_scores[
                    query_index * keypoint_count + train_index];
            if (value > best_value) {
                best_value = value;
                best = query_index;
            }
        }
        train_to_query[train_index] = best;
    }

    std::vector<vision::FeatureMatch> matches;
    for (int query_index = 0;
        query_index < keypoint_count;
        ++query_index) {
        const int train_index = query_to_train[query_index];
        const float score = std::exp(query_scores[query_index]);
        if (train_to_query[train_index] != query_index ||
            !std::isfinite(score) ||
            score <= filter_threshold) {
            continue;
        }
        vision::FeatureMatch match;
        match.query_index = query_index;
        match.train_index = train_index;
        match.query_point = query.keypoints[query_index];
        match.train_point = train.keypoints[train_index];
        match.score = score;
        matches.push_back(std::move(match));
    }
    return matches;
}

LightGlueMatcher::LightGlueMatcher(
    const std::string& model_path,
    std::string feature_type,
    int num_keypoints,
    int descriptor_dim,
    float filter_threshold,
    int num_threads,
    bool lazy_load,
    std::string provider)
    : BaseModel(model_path, lazy_load),
        feature_type_(std::move(feature_type)),
        num_keypoints_(num_keypoints),
        descriptor_dim_(descriptor_dim),
        filter_threshold_(filter_threshold),
        num_threads_(num_threads),
        provider_(std::move(provider)) {
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel> LightGlueMatcher::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for LightGlueMatcher");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<LightGlueMatcher>(
        model_path,
        vision_core::yaml_utils::getString(
            params, "feature_type", "superpoint"),
        vision_core::yaml_utils::getInt(
            params, "num_keypoints", 512),
        vision_core::yaml_utils::getInt(
            params, "descriptor_dim", 256),
        vision_core::yaml_utils::getFloat(
            params, "filter_threshold", 0.1f),
        vision_core::yaml_utils::getInt(params, "num_threads", 4),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void LightGlueMatcher::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (session_->GetInputCount() != 2 || output_num_ != 1) {
        throw std::runtime_error(
            "LightGlueMatcher expects two inputs and one output");
    }
    const auto keypoint_shape =
        session_->GetInputTypeInfo(0)
            .GetTensorTypeAndShapeInfo().GetShape();
    const auto descriptor_shape =
        session_->GetInputTypeInfo(1)
            .GetTensorTypeAndShapeInfo().GetShape();
    if (keypoint_shape.size() != 3 ||
        descriptor_shape.size() != 3 ||
        keypoint_shape[0] != 2 ||
        keypoint_shape[1] != num_keypoints_ ||
        keypoint_shape[2] != 2 ||
        descriptor_shape[0] != 2 ||
        descriptor_shape[1] != num_keypoints_ ||
        descriptor_shape[2] != descriptor_dim_) {
        throw std::runtime_error(
            "LightGlueMatcher config is incompatible with ONNX input shapes");
    }
    model_loaded_ = true;
}

std::vector<vision::FeatureMatch> LightGlueMatcher::match(
    const vision_core::LocalFeaturePairInput& input) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto total_begin = std::chrono::steady_clock::now();
    const std::string query_error = validate_lightglue_features(
        input.query,
        feature_type_,
        num_keypoints_,
        descriptor_dim_);
    if (!query_error.empty()) {
        throw std::invalid_argument(
            "LightGlue query features: " + query_error);
    }
    const std::string train_error = validate_lightglue_features(
        input.train,
        feature_type_,
        num_keypoints_,
        descriptor_dim_);
    if (!train_error.empty()) {
        throw std::invalid_argument(
            "LightGlue train features: " + train_error);
    }

    const auto preprocess_begin = std::chrono::steady_clock::now();
    std::vector<float> normalized_keypoints(
        static_cast<size_t>(2) * num_keypoints_ * 2);
    for (int i = 0; i < num_keypoints_; ++i) {
        normalized_keypoints[i * 2] =
            2.0f * input.query.keypoints[i].x /
                input.query.image_width -
            1.0f;
        normalized_keypoints[i * 2 + 1] =
            2.0f * input.query.keypoints[i].y /
                input.query.image_height -
            1.0f;
        const size_t train_offset =
            static_cast<size_t>(num_keypoints_) * 2 + i * 2;
        normalized_keypoints[train_offset] =
            2.0f * input.train.keypoints[i].x /
                input.train.image_width -
            1.0f;
        normalized_keypoints[train_offset + 1] =
            2.0f * input.train.keypoints[i].y /
                input.train.image_height -
            1.0f;
    }
    std::vector<float> descriptors;
    descriptors.reserve(
        static_cast<size_t>(2) *
        num_keypoints_ *
        descriptor_dim_);
    descriptors.insert(
        descriptors.end(),
        input.query.descriptors.begin(),
        input.query.descriptors.end());
    descriptors.insert(
        descriptors.end(),
        input.train.descriptors.begin(),
        input.train.descriptors.end());
    const auto preprocess_end = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(
        std::chrono::duration<double, std::milli>(
            preprocess_end - preprocess_begin).count());

    const std::array<int64_t, 3> keypoint_shape{
        2, num_keypoints_, 2};
    const std::array<int64_t, 3> descriptor_shape{
        2, num_keypoints_, descriptor_dim_};
    std::vector<Ort::Value> tensors;
    tensors.reserve(2);
    tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        normalized_keypoints.data(),
        normalized_keypoints.size(),
        keypoint_shape.data(),
        keypoint_shape.size()));
    tensors.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        descriptors.data(),
        descriptors.size(),
        descriptor_shape.data(),
        descriptor_shape.size()));

    const auto infer_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        tensors.data(),
        tensors.size(),
        output_node_names_.data(),
        output_node_names_.size());
    const auto infer_end = std::chrono::steady_clock::now();
    const double infer_ms =
        std::chrono::duration<double, std::milli>(
            infer_end - infer_begin).count();
    set_runtime_model_infer_ms(infer_ms);
    add_runtime_component_timing("lightglue.infer", infer_ms);

    const auto postprocess_begin = std::chrono::steady_clock::now();
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error(
            "LightGlueMatcher returned invalid output");
    }
    const auto output_info =
        outputs[0].GetTensorTypeAndShapeInfo();
    const auto output_shape = output_info.GetShape();
    if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_shape.size() != 3 ||
        output_shape[0] != 1 ||
        output_shape[1] != num_keypoints_ ||
        output_shape[2] != num_keypoints_) {
        throw std::runtime_error(
            "LightGlueMatcher expects output shape [1,N,N] float32");
    }
    std::vector<vision::FeatureMatch> result =
        filter_lightglue_matches(
            outputs[0].GetTensorData<float>(),
            num_keypoints_,
            input.query,
            input.train,
            filter_threshold_);
    const auto postprocess_end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_begin).count());
    set_runtime_total_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - total_begin).count());
    return result;
}

vision_core::InferResponse LightGlueMatcher::Run(
    const vision_core::InferRequest& request) {
    vision_core::InferResponse response;
    if (request.intent !=
        vision_core::InferIntent::kMatchLocalFeatures) {
        response.ok = false;
        response.error_message =
            "LightGlueMatcher only supports kMatchLocalFeatures";
        return response;
    }
    const auto* pair =
        std::get_if<vision_core::LocalFeaturePairInput>(
            &request.input);
    if (pair == nullptr) {
        response.ok = false;
        response.error_message =
            "LightGlueMatcher expects LocalFeaturePairInput";
        return response;
    }
    std::vector<vision::FeatureMatch> matches = match(*pair);
    response.results.reserve(matches.size());
    for (auto& item : matches) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::InferIntent>
LightGlueMatcher::supported_intents() const {
    return {vision_core::InferIntent::kMatchLocalFeatures};
}

static vision_core::ModelRegistrar<LightGlueMatcher> registrar(
    "LightGlueMatcher");

}  // namespace vision_deploy
