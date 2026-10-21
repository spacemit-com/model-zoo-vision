/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "emotion_recognizer.h"

#include <cassert>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "common.h"
#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> EmotionRecognizer::create(const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(config, {"feature_mode"});
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for EmotionRecognizer");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for EmotionRecognizer");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    bool feature_mode = vision_core::yaml_utils::getBool(default_params, "feature_mode", false);
    return std::make_unique<EmotionRecognizer>(model_path, num_threads, lazy_load, provider, feature_mode);
}

EmotionRecognizer::EmotionRecognizer(const std::string& model_path,
                                    int num_threads,
                                    bool lazy_load,
                                    const std::string& provider,
                                    bool feature_mode)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        target_size_(224, 224),
        provider_(provider),
        feature_mode_(feature_mode) {
    if (!lazy_load) {
        load_model();
    }
}

void EmotionRecognizer::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat EmotionRecognizer::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = target_size_.width;
    spec.output_height = target_size_.height;
    spec.output_rgb = false;
    spec.interpolation = vision_operators::PreprocessInterpolation::kNearest;
    spec.mean = {91.4953F, 103.8827F, 131.0912F};
    return vision_operators::preprocess_bgr_to_nchw(image, spec);
}

vision_core::BaseModel::PreparedImage
EmotionRecognizer::prepare_input(
    const vision_core::ImageInput& input) {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = target_size_.width;
    spec.output_height = target_size_.height;
    spec.output_rgb = false;
    spec.interpolation =
        vision_operators::PreprocessInterpolation::kNearest;
    spec.mean = {91.4953F, 103.8827F, 131.0912F};
    return prepare_image(
        input, spec,
        [this](const cv::Mat& bgr) {
            return preprocess(bgr);
        });
}

vision_common::ClassificationResultList
EmotionRecognizer::classify_input(
    const vision_core::ImageInput& input, int top_k) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    // Preprocess
    const auto t_pre0 = std::chrono::steady_clock::now();
    auto prepared = prepare_input(input);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    // Run inference using base class method
    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session(prepared.tensor());
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    // Postprocess
    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::ClassificationResultList results = postprocess(outputs, top_k);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}


vision_common::EmbeddingResult
EmotionRecognizer::infer_embedding_input(
    const vision_core::ImageInput& input) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const auto t_pre0 = std::chrono::steady_clock::now();
    auto prepared = prepare_input(input);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session(prepared.tensor());
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    // Feature mode: return raw output vector (no argmax, no L2 normalization).
    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::EmbeddingResult result;
    result.embedding = read_output_scores(outputs);
    result.score = 1.0f;
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return result;
}


std::vector<vision_core::InferIntent> EmotionRecognizer::supported_intents() const {
    if (feature_mode_) {
        return {vision_core::InferIntent::kEmbed};
    }
    return {vision_core::InferIntent::kClassify};
}

vision_core::InferResponse EmotionRecognizer::RunImpl(const vision_core::InferRequest& request) {
    const auto& image_input = std::get<vision_core::ImageInput>(request.input);

    vision_core::InferResponse response;
    if (request.intent == vision_core::InferIntent::kEmbed) {
        response.results.emplace_back(
            infer_embedding_input(image_input));
        return response;
    }

    // Default: classification
    vision_common::ClassificationResultList task_results =
        classify_input(image_input, request.params.top_k < 0 ? 1 : request.params.top_k);
    return vision_core::make_infer_response(std::move(task_results));
}

std::vector<vision_core::ModelCapability> EmotionRecognizer::get_capabilities() const {
    return {};
}

vision_common::ClassificationResultList EmotionRecognizer::postprocess(
    std::vector<Ort::Value>& outputs, int top_k) {
    // Preserve raw model scores and the default top-1 behavior.
    return vision_common::build_classification_top_k(read_output_scores(outputs), top_k, false);
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<EmotionRecognizer> registrar("EmotionRecognizer");

}  // namespace vision_deploy
