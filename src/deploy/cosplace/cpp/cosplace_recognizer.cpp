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
#include "embedding_utils.h"
#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {

// ImageNet normalization constants
constexpr float kImageNetMean[] = {0.485f, 0.456f, 0.406f};
constexpr float kImageNetStd[] = {0.229f, 0.224f, 0.225f};

// Resize to the model input, BGR->RGB, (x / 255 - mean) / std.
vision_operators::ImagePreprocessSpec make_preprocess_spec(int width, int height) {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = width;
    spec.output_height = height;
    spec.output_rgb = true;
    spec.mean = {
        kImageNetMean[0] * 255.0f,
        kImageNetMean[1] * 255.0f,
        kImageNetMean[2] * 255.0f
    };
    spec.scale = {
        1.0f / (kImageNetStd[0] * 255.0f),
        1.0f / (kImageNetStd[1] * 255.0f),
        1.0f / (kImageNetStd[2] * 255.0f)
    };
    return spec;
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> CosPlaceRecognizer::create(const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(config, {});
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for CosPlaceRecognizer");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for CosPlaceRecognizer");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    if (num_threads <= 0) {
        throw std::runtime_error("default_params.num_threads must be positive for CosPlaceRecognizer");
    }
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
    // Preprocessing reads H/W from the NCHW input shape; reject models with an
    // unexpected rank or dynamic spatial dims instead of indexing out of range.
    if (input_shape_.size() != 4 || input_shape_[2] <= 0 || input_shape_[3] <= 0) {
        throw std::runtime_error(
            "CosPlaceRecognizer expects a static NCHW input shape: " + model_path_);
    }
    model_loaded_ = true;
}

cv::Mat CosPlaceRecognizer::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    const auto spec = make_preprocess_spec(
        static_cast<int>(input_shape_[3]), static_cast<int>(input_shape_[2]));
    return vision_operators::preprocess_bgr_to_nchw(image, spec);
}

vision_common::EmbeddingResult
CosPlaceRecognizer::infer_embedding_input(
    const vision_core::ImageInput& input) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    // Preprocess
    const auto t_pre0 = std::chrono::steady_clock::now();
    const auto spec = make_preprocess_spec(
        static_cast<int>(input_shape_[3]), static_cast<int>(input_shape_[2]));
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

vision_core::InferResponse CosPlaceRecognizer::RunImpl(const vision_core::InferRequest& request) {
    const auto& image_input = std::get<vision_core::ImageInput>(request.input);
    vision_core::InferResponse response;
    response.results.emplace_back(
        infer_embedding_input(image_input));
    return response;
}

std::vector<vision_core::ModelCapability> CosPlaceRecognizer::get_capabilities() const {
    // ModelCapability only describes optional extras (currently just kDraw).
    // Embedding support is advertised through supported_intents() (kEmbed);
    // a place descriptor has nothing to draw, matching ArcFace/AdaFace.
    return {};
}

std::vector<float> CosPlaceRecognizer::postprocess(std::vector<Ort::Value>& outputs) {
    if (outputs.empty()) {
        throw std::runtime_error("CosPlaceRecognizer::postprocess: no output tensors returned");
    }

    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("CosPlaceRecognizer::postprocess: expected a float32 output tensor");
    }
    std::vector<int64_t> dims = tensor_info.GetShape();

    // Use all dimensions after the batch dimension (a 1-D output is treated as
    // the full descriptor, so no batch dimension is stripped).
    size_t embedding_size = 1;
    for (size_t i = (dims.size() > 1 ? 1 : 0); i < dims.size(); ++i) {
        if (dims[i] <= 0) {
            throw std::runtime_error("CosPlaceRecognizer::postprocess: invalid output shape");
        }
        embedding_size *= static_cast<size_t>(dims[i]);
    }
    if (dims.empty() || embedding_size > tensor_info.GetElementCount()) {
        throw std::runtime_error("CosPlaceRecognizer::postprocess: output tensor is smaller than expected");
    }

    const float* output_data = outputs[0].GetTensorMutableData<float>();
    if (output_data == nullptr) {
        throw std::runtime_error("CosPlaceRecognizer::postprocess: null output data");
    }

    // Copy embedding data
    std::vector<float> embedding(output_data, output_data + embedding_size);

    // L2 normalize the descriptor
    embedding = vision_common::normalize_embedding(embedding);

    return embedding;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<CosPlaceRecognizer> registrar("CosPlaceRecognizer");

}  // namespace vision_deploy
