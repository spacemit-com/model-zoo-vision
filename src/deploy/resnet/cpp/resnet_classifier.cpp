/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "resnet_classifier.h"

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

std::unique_ptr<vision_core::BaseModel> ResNetClassifier::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for ResNetClassifier");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for ResNetClassifier");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    return std::make_unique<ResNetClassifier>(model_path, num_threads, lazy_load, provider);
}

ResNetClassifier::ResNetClassifier(const std::string& model_path,
                                    int num_threads,
                                    bool lazy_load,
                                    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        resize_size_(256, 256),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void ResNetClassifier::load_model() {
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat ResNetClassifier::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);  // width
    int inputHeight = static_cast<int>(input_shape_[2]);  // height

    // Use common preprocess_classification function
    // Same as Python: preprocess_classification(image, input_shape, resize_size=(256,256))
    cv::Mat blob = vision_common::preprocess_classification(
        image,
        std::make_pair(inputHeight, inputWidth),
        cv::Scalar(0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f),  // ImageNet mean
        cv::Scalar(0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f),  // ImageNet std
        resize_size_,
        true);  // center_crop

    return blob;
}

std::vector<vision_common::Result> ResNetClassifier::classify(const cv::Mat& image) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

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
    std::vector<vision_common::Result> results = postprocess(outputs);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}

std::vector<vision_core::ModelCapability> ResNetClassifier::get_capabilities() const {
    return {vision_core::ModelCapability::kImageInput};
}

std::vector<vision_common::Result> ResNetClassifier::postprocess(std::vector<Ort::Value>& outputs) {
    // Extract logits from outputs
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = tensor_info.GetShape();

    size_t num_classes = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        num_classes *= dims[i];
    }

    // Copy class scores
    std::vector<float> class_scores(output_data, output_data + num_classes);

    // Apply softmax (same as Python: outputs are logits, need softmax)
    // Find max for numerical stability
    float max_score = *std::max_element(class_scores.begin(), class_scores.end());
    float exp_sum = 0.0f;
    for (float& score : class_scores) {
        score = std::exp(score - max_score);
        exp_sum += score;
    }
    for (float& score : class_scores) {
        score /= exp_sum;
    }

    // Find top class
    int top_class = std::max_element(class_scores.begin(), class_scores.end()) - class_scores.begin();
    float top_score = class_scores[top_class];

    // Create Result object
    vision_common::Result result;
    result.x1 = 0.0f;
    result.y1 = 0.0f;
    result.x2 = 0.0f;
    result.y2 = 0.0f;
    result.score = top_score;
    result.label = top_class;
    result.class_scores = class_scores;

    return std::vector<vision_common::Result>{result};
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<ResNetClassifier> registrar("ResNetClassifier");

}  // namespace vision_deploy

