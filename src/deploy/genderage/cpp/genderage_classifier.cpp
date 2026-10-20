/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "genderage_classifier.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
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

std::unique_ptr<vision_core::BaseModel> GenderAgeClassifier::create(const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(config, {"input_mean", "input_std"});
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for GenderAgeClassifier");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for GenderAgeClassifier");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    float input_mean = vision_core::yaml_utils::getFloat(default_params, "input_mean", 127.5f);
    float input_std = vision_core::yaml_utils::getFloat(default_params, "input_std", 128.0f);
    return std::make_unique<GenderAgeClassifier>(
        model_path, num_threads, lazy_load, provider, input_mean, input_std);
}

GenderAgeClassifier::GenderAgeClassifier(const std::string& model_path,
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
    vision_common::validate_classification_normalization(
        cv::Scalar::all(input_mean_), cv::Scalar::all(input_std_));
    if (!lazy_load) {
        load_model();
    }
}

void GenderAgeClassifier::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat GenderAgeClassifier::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();

    const float inv_std = 1.0f / input_std_;
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = target_size_.width;
    spec.output_height = target_size_.height;
    spec.mean = {input_mean_, input_mean_, input_mean_};
    spec.scale = {inv_std, inv_std, inv_std};
    return vision_operators::preprocess_bgr_to_nchw(image, spec);
}

vision_common::ClassificationResultList GenderAgeClassifier::classify(const cv::Mat& image) {
    vision_core::ImageInput input;
    input.image = image;
    return classify_input(input);
}

vision_common::ClassificationResultList
GenderAgeClassifier::classify_input(
    const vision_core::ImageInput& input) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const auto t_pre0 = std::chrono::steady_clock::now();
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = target_size_.width;
    spec.output_height = target_size_.height;
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
    vision_common::ClassificationResultList results = postprocess(outputs);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

std::vector<vision_core::InferIntent> GenderAgeClassifier::supported_intents() const {
    return {vision_core::InferIntent::kClassify};
}

vision_core::InferResponse GenderAgeClassifier::Run(const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kClassify) {
        return unsupported_intent_response(request.intent);
    }
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "GenderAgeClassifier expects ImageInput";
        return response;
    }

    vision_common::ClassificationResultList task_results =
        classify_input(*image_input);
    vision_core::InferResponse response;
    response.results.reserve(task_results.size());
    for (auto& item : task_results) {
        response.results.emplace_back(std::move(item));
    }
    return response;
}

std::vector<vision_core::ModelCapability> GenderAgeClassifier::get_capabilities() const {
    return {};
}

vision_common::ClassificationResultList GenderAgeClassifier::postprocess(std::vector<Ort::Value>& outputs) {
    const auto gender = read_output_scores(outputs);
    vision_common::ClassificationResult result;
    int age = 0;
    if (outputs.size() == 1) {
        if (gender.size() != 3) {
            throw std::runtime_error("GenderAge output must contain female, male and age scores");
        }
        // Clamp before rounding so finite but very large age values cannot overflow lround.
        age = static_cast<int>(std::lround(std::clamp(gender[2], 0.01f, 1.0f) * 100.0f));
    } else {
        if (gender.size() != 2) {
            throw std::runtime_error("GenderAge gender output must contain exactly 2 scores");
        }
        const auto age_scores = read_output_scores(outputs, 1);
        age = static_cast<int>(std::max_element(age_scores.begin(), age_scores.end()) - age_scores.begin());
    }
    result.label = gender[1] > gender[0] ? 1 : 0;
    result.score = std::max(gender[0], gender[1]);
    result.class_scores = {gender[0], gender[1], static_cast<float>(age)};
    return {result};
}

static vision_core::ModelRegistrar<GenderAgeClassifier> registrar("GenderAgeClassifier");

}  // namespace vision_deploy
