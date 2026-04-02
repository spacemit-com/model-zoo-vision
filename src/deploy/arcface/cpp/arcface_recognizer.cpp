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

    // Convert BGR to RGB (same as Python: cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    // Resize with letterbox using BICUBIC interpolation (same as Python PIL Image.BICUBIC)
    // Python: scale = min(w / iw, h / ih), then resize with BICUBIC
    int orig_h = rgb_image.rows;
    int orig_w = rgb_image.cols;

    // Calculate scale ratio (same as Python: scale = min(w / iw, h / ih))
    float scale = std::min(static_cast<float>(inputWidth) / orig_w,
                            static_cast<float>(inputHeight) / orig_h);

    // Compute new dimensions (same as Python: nw = int(iw * scale), nh = int(ih * scale))
    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);

    // Resize with BICUBIC interpolation (same as Python PIL Image.BICUBIC)
    cv::Mat resized;
    cv::resize(rgb_image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    // Add padding (same as Python: new_image.paste(image, ((w - nw) // 2, (h - nh) // 2)))
    // Python uses integer division, so we use the same
    int pad_w = (inputWidth - new_w) / 2;
    int pad_h = (inputHeight - new_h) / 2;

    // Create padded image with gray background (128, 128, 128)
    cv::Mat padded = cv::Mat::zeros(inputHeight, inputWidth, rgb_image.type());
    padded.setTo(cv::Scalar(128, 128, 128));

    // Paste resized image in the center (same as Python: new_image.paste(image, (pad_w, pad_h)))
    cv::Rect roi(pad_w, pad_h, new_w, new_h);
    resized.copyTo(padded(roi));

    // Convert to float and normalize to [-1, 1].
    // Use convertTo(scale, shift) to avoid OpenCV multi-channel MatExpr warnings.
    cv::Mat normalized;
    padded.convertTo(normalized, CV_32F, 1.0 / 127.5, -1.0);

    // HWC to CHW and add batch dimension (same as Python: np.transpose(img, (2, 0, 1)))
    cv::Mat blob = cv::dnn::blobFromImage(normalized, 1.0,
                                        cv::Size(inputWidth, inputHeight),
                                        cv::Scalar(0, 0, 0), false, false, CV_32F);

    return blob;
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

