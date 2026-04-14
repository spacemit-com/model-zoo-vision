/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "emotion_recognizer.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> EmotionRecognizer::create(const YAML::Node& config, bool lazy_load) {
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
    return std::make_unique<EmotionRecognizer>(model_path, num_threads, lazy_load, provider);
}

EmotionRecognizer::EmotionRecognizer(const std::string& model_path,
                                    int num_threads,
                                    bool lazy_load,
                                    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        target_size_(224, 224),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void EmotionRecognizer::load_model() {
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat EmotionRecognizer::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    // Resize to target size (224, 224)
    cv::Mat resized;
    cv::resize(image, resized, target_size_, 0, 0, cv::INTER_NEAREST);

    // blobFromImage: float conversion, mean subtraction, and HWC->CHW in one call
    // swapRB=false because emotion model expects BGR order
    return cv::dnn::blobFromImage(resized, 1.0, target_size_,
                                    cv::Scalar(91.4953, 103.8827, 131.0912),
                                    false, false, CV_32F);
}

std::vector<vision_common::Result> EmotionRecognizer::classify(const cv::Mat& image) {
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

std::vector<vision_core::ModelCapability> EmotionRecognizer::get_capabilities() const {
    return {vision_core::ModelCapability::kImageInput};
}

std::vector<vision_common::Result> EmotionRecognizer::postprocess(std::vector<Ort::Value>& outputs) {
    // Check if outputs is valid
    if (outputs.empty()) {
        throw std::runtime_error("Postprocess: outputs is empty");
    }

    // Extract logits from outputs
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = tensor_info.GetShape();

    if (dims.empty()) {
        throw std::runtime_error("Postprocess: output tensor has no dimensions");
    }

    // Calculate total number of elements (skip batch dimension)
    size_t num_classes = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        if (dims[i] <= 0) {
            throw std::runtime_error("Postprocess: invalid dimension size");
        }
        num_classes *= static_cast<size_t>(dims[i]);
    }

    if (num_classes == 0) {
        throw std::runtime_error("Postprocess: num_classes is zero");
    }

    // Get output data
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    if (output_data == nullptr) {
        throw std::runtime_error("Postprocess: output_data is null");
    }

    // Copy class scores
    std::vector<float> class_scores(output_data, output_data + num_classes);

    if (class_scores.empty()) {
        throw std::runtime_error("Postprocess: class_scores is empty");
    }

    // Find emotion class (argmax) - same as Python: np.argmax(output)
    auto max_it = std::max_element(class_scores.begin(), class_scores.end());
    if (max_it == class_scores.end()) {
        throw std::runtime_error("Postprocess: failed to find max element");
    }
    int emotion_class = static_cast<int>(max_it - class_scores.begin());
    float emotion_score = *max_it;

    // Create Result object
    vision_common::Result result;
    result.x1 = 0.0f;
    result.y1 = 0.0f;
    result.x2 = 0.0f;
    result.y2 = 0.0f;
    result.score = emotion_score;
    result.label = emotion_class;
    result.class_scores = class_scores;

    return std::vector<vision_common::Result>{result};
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<EmotionRecognizer> registrar("EmotionRecognizer");

}  // namespace vision_deploy

