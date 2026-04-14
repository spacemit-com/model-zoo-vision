/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arcface_recognizer.h"

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

std::unique_ptr<vision_core::BaseModel> ArcFaceRecognizer::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for ArcFaceRecognizer");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for ArcFaceRecognizer");
    }

    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<ArcFaceRecognizer>(model_path, num_threads, lazy_load, provider);
}

ArcFaceRecognizer::ArcFaceRecognizer(const std::string& model_path,
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

void ArcFaceRecognizer::load_model() {
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat ArcFaceRecognizer::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);  // width
    int inputHeight = static_cast<int>(input_shape_[2]);  // height

    // Resize with letterbox using BICUBIC interpolation on uint8 first
    int orig_h = image.rows;
    int orig_w = image.cols;

    float scale = std::min(static_cast<float>(inputWidth) / orig_w,
                            static_cast<float>(inputHeight) / orig_h);

    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    int pad_w = (inputWidth - new_w) / 2;
    int pad_h = (inputHeight - new_h) / 2;

    // Create padded image with gray background (128, 128, 128) in BGR
    cv::Mat padded(inputHeight, inputWidth, image.type(), cv::Scalar(128, 128, 128));
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));

    // blobFromImage: BGR->RGB (swapRB=true), float conversion, scale 1/127.5,
    // shift -1.0 to normalize to [-1, 1], and HWC->CHW — all in one call
    return cv::dnn::blobFromImage(padded, 1.0 / 127.5,
                                    cv::Size(inputWidth, inputHeight),
                                    cv::Scalar(127.5, 127.5, 127.5),
                                    true, false, CV_32F);
}

std::vector<float> ArcFaceRecognizer::infer_embedding(const cv::Mat& image) {
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
    std::vector<float> embedding = postprocess(outputs);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return embedding;
}

std::vector<vision_core::ModelCapability> ArcFaceRecognizer::get_capabilities() const {
    return {
        vision_core::ModelCapability::kImageInput,
        vision_core::ModelCapability::kEmbedding};
}

std::vector<float> ArcFaceRecognizer::postprocess(std::vector<Ort::Value>& outputs) {
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

    // L2 normalize using common function
    embedding = vision_common::normalize_embedding(embedding);

    return embedding;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<ArcFaceRecognizer> registrar("ArcFaceRecognizer");

}  // namespace vision_deploy

