/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cosplace_recognizer.h"

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

// ImageNet normalization constants
static const float IMAGENET_MEAN[] = {0.485f, 0.456f, 0.406f};
static const float IMAGENET_STD[] = {0.229f, 0.224f, 0.225f};

std::unique_ptr<vision_core::BaseModel> CosPlaceRecognizer::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for CosPlaceRecognizer");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for CosPlaceRecognizer");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<CosPlaceRecognizer>(model_path, num_threads, lazy_load, provider);
}

CosPlaceRecognizer::CosPlaceRecognizer(const std::string& model_path,
        int num_threads,
        bool lazy_load,
        const std::string& provider)
        : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void CosPlaceRecognizer::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat CosPlaceRecognizer::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);   // width
    int inputHeight = static_cast<int>(input_shape_[2]);  // height

    // CosPlace uses ImageNet normalization: resize, BGR->RGB, (x/255 - mean) / std
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = inputWidth;
    spec.output_height = inputHeight;
    spec.output_rgb = true;
    spec.mean = {
        IMAGENET_MEAN[0] * 255.0f,
        IMAGENET_MEAN[1] * 255.0f,
        IMAGENET_MEAN[2] * 255.0f
    };
    spec.scale = {
        1.0f / (IMAGENET_STD[0] * 255.0f),
        1.0f / (IMAGENET_STD[1] * 255.0f),
        1.0f / (IMAGENET_STD[2] * 255.0f)
    };
    return vision_operators::preprocess_bgr_to_nchw(image, spec);
}

vision_common::EmbeddingResult CosPlaceRecognizer::infer_embedding(const cv::Mat& image) {
    vision_core::ImageInput input;
    input.image = image;
    return infer_embedding_input(input);
}

vision_common::EmbeddingResult
CosPlaceRecognizer::infer_embedding_input(
    const vision_core::ImageInput& input) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    // Preprocess
    const auto t_pre0 = std::chrono::steady_clock::now();
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = static_cast<int>(input_shape_[3]);
    spec.output_height = static_cast<int>(input_shape_[2]);
    spec.output_rgb = true;
    spec.mean = {
        IMAGENET_MEAN[0] * 255.0f,
        IMAGENET_MEAN[1] * 255.0f,
        IMAGENET_MEAN[2] * 255.0f
    };
    spec.scale = {
        1.0f / (IMAGENET_STD[0] * 255.0f),
        1.0f / (IMAGENET_STD[1] * 255.0f),
        1.0f / (IMAGENET_STD[2] * 255.0f)
    };
    auto prepared = prepare_image(
        input, spec,
        [this](const cv::Mat& bgr) {
            return preprocess(bgr);
        });
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
    std::vector<float> embedding = postprocess(outputs);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Create EmbeddingResult
    vision_common::EmbeddingResult result;
    result.embedding = std::move(embedding);
    result.score = 1.0f;  // Quality score

    return result;
}

std::vector<vision_core::InferIntent> CosPlaceRecognizer::supported_intents() const {
    return {vision_core::InferIntent::kEmbed};
}

vision_core::InferResponse CosPlaceRecognizer::Run(const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kEmbed) {
        return unsupported_intent_response(request.intent);
    }
    const auto* image_input = std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "CosPlaceRecognizer expects ImageInput";
        return response;
    }
    vision_core::InferResponse response;
    response.results.emplace_back(
        infer_embedding_input(*image_input));
    return response;
}

std::vector<vision_core::ModelCapability> CosPlaceRecognizer::get_capabilities() const {
    return {};
}

std::vector<float> CosPlaceRecognizer::postprocess(std::vector<Ort::Value>& outputs) {
    // Extract embedding from outputs
    const float* output_data = outputs[0].GetTensorMutableData<float>();
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = tensor_info.GetShape();

    size_t embedding_size = 1;
    for (size_t i = 1; i < dims.size(); ++i) {
        embedding_size *= dims[i];
    }

    // Copy embedding data
    std::vector<float> embedding(output_data, output_data + embedding_size);

    // L2 normalize the descriptor
    embedding = vision_common::normalize_embedding(embedding);

    return embedding;
}

float CosPlaceRecognizer::compute_distance(const std::vector<float>& embedding1,
        const std::vector<float>& embedding2) {
    if (embedding1.size() != embedding2.size()) {
        throw std::runtime_error("Embeddings must have the same dimension");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < embedding1.size(); ++i) {
        float diff = embedding1[i] - embedding2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<CosPlaceRecognizer> registrar("CosPlaceRecognizer");

}  // namespace vision_deploy
