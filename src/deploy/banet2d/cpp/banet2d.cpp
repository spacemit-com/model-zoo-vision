/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "banet2d.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

BANetLetterbox make_banet_letterbox(
    int input_width,
    int input_height,
    int output_width,
    int output_height) {
    if (input_width <= 0 || input_height <= 0 ||
        output_width <= 0 || output_height <= 0) {
        throw std::invalid_argument(
            "BANet2D letterbox dimensions must be positive");
    }
    const float ratio = std::min(
        static_cast<float>(output_height) / input_height,
        static_cast<float>(output_width) / input_width);
    BANetLetterbox geometry;
    geometry.input_width = input_width;
    geometry.input_height = input_height;
    geometry.output_width = output_width;
    geometry.output_height = output_height;
    geometry.resized_width =
        std::max(1, static_cast<int>(std::round(input_width * ratio)));
    geometry.resized_height =
        std::max(1, static_cast<int>(std::round(input_height * ratio)));
    geometry.pad_left =
        (output_width - geometry.resized_width) / 2;
    geometry.pad_top =
        (output_height - geometry.resized_height) / 2;
    geometry.pad_right =
        output_width - geometry.resized_width - geometry.pad_left;
    geometry.pad_bottom =
        output_height - geometry.resized_height - geometry.pad_top;
    return geometry;
}

cv::Mat restore_banet_disparity(
    const cv::Mat& model_disparity,
    const BANetLetterbox& geometry,
    const cv::Size& original_size) {
    if (model_disparity.empty() ||
        model_disparity.type() != CV_32FC1) {
        throw std::invalid_argument(
            "BANet2D disparity must be non-empty CV_32FC1");
    }
    if (original_size.width <= 0 || original_size.height <= 0 ||
        geometry.resized_width <= 0 ||
        geometry.resized_height <= 0) {
        throw std::invalid_argument(
            "BANet2D restore dimensions must be positive");
    }
    const cv::Rect valid(
        geometry.pad_left,
        geometry.pad_top,
        geometry.resized_width,
        geometry.resized_height);
    if (valid.x < 0 || valid.y < 0 ||
        valid.x + valid.width > model_disparity.cols ||
        valid.y + valid.height > model_disparity.rows) {
        throw std::runtime_error(
            "BANet2D disparity crop exceeds model output");
    }
    cv::Mat restored;
    cv::resize(
        model_disparity(valid),
        restored,
        original_size,
        0.0,
        0.0,
        cv::INTER_LINEAR);
    restored *=
        static_cast<float>(original_size.width) /
        static_cast<float>(geometry.resized_width);
    return restored;
}

BANet2D::BANet2D(
    const std::string& model_path,
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

std::unique_ptr<vision_core::BaseModel> BANet2D::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for BANet2D");
    }
    const YAML::Node params = config["default_params"];
    const int num_threads =
        vision_core::yaml_utils::getInt(params, "num_threads", 4);
    return std::make_unique<BANet2D>(
        model_path,
        num_threads,
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void BANet2D::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_shape_.size() != 4 ||
        input_shape_[0] != 2 ||
        input_shape_[1] != 3 ||
        input_shape_[2] <= 0 ||
        input_shape_[3] <= 0) {
        throw std::runtime_error(
            "BANet2D expects input shape [2,3,H,W]");
    }
    if (output_num_ != 1) {
        throw std::runtime_error(
            "BANet2D expects exactly one disparity output");
    }
    model_loaded_ = true;
}

cv::Mat BANet2D::preprocess_one(
    const cv::Mat& bgr,
    const BANetLetterbox& geometry) const {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument(
            "BANet2D expects a non-empty BGR8 image");
    }
    cv::Mat resized;
    cv::resize(
        bgr,
        resized,
        cv::Size(
            geometry.resized_width,
            geometry.resized_height),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        geometry.pad_top,
        geometry.pad_bottom,
        geometry.pad_left,
        geometry.pad_right,
        cv::BORDER_REPLICATE);
    return cv::dnn::blobFromImage(
        padded,
        1.0,
        cv::Size(),
        cv::Scalar(),
        true,
        false,
        CV_32F);
}

vision::Disparity BANet2D::infer_stereo(
    const vision_core::StereoImageInput& input) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto total_begin = std::chrono::steady_clock::now();

    const auto preprocess_begin = std::chrono::steady_clock::now();
    const cv::Mat left =
        vision_operators::image_input_to_bgr_cpu(input.left);
    const cv::Mat right =
        vision_operators::image_input_to_bgr_cpu(input.right);
    if (left.size() != right.size()) {
        throw std::invalid_argument(
            "BANet2D left and right image dimensions must match");
    }
    const int model_height = static_cast<int>(input_shape_[2]);
    const int model_width = static_cast<int>(input_shape_[3]);
    const BANetLetterbox geometry = make_banet_letterbox(
        left.cols,
        left.rows,
        model_width,
        model_height);
    const cv::Mat left_blob = preprocess_one(left, geometry);
    const cv::Mat right_blob = preprocess_one(right, geometry);
    const size_t image_values =
        static_cast<size_t>(3) * model_height * model_width;
    cv::Mat batch(
        1,
        static_cast<int>(image_values * 2),
        CV_32FC1);
    std::memcpy(
        batch.ptr<float>(),
        left_blob.ptr<float>(),
        image_values * sizeof(float));
    std::memcpy(
        batch.ptr<float>() + image_values,
        right_blob.ptr<float>(),
        image_values * sizeof(float));
    const auto preprocess_end = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(
        std::chrono::duration<double, std::milli>(
            preprocess_end - preprocess_begin).count());

    const auto infer_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(batch);
    const auto infer_end = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(
        std::chrono::duration<double, std::milli>(
            infer_end - infer_begin).count());

    const auto postprocess_begin = std::chrono::steady_clock::now();
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error(
            "BANet2D returned an invalid disparity output");
    }
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    if (info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 4 ||
        shape[0] != 1 ||
        shape[1] != 1 ||
        shape[2] <= 0 ||
        shape[3] <= 0) {
        throw std::runtime_error(
            "BANet2D expects output shape [1,1,H,W] float32");
    }
    const int output_height = static_cast<int>(shape[2]);
    const int output_width = static_cast<int>(shape[3]);
    cv::Mat model_map(
        output_height,
        output_width,
        CV_32FC1,
        const_cast<float*>(outputs[0].GetTensorData<float>()));
    if (output_width != geometry.output_width ||
        output_height != geometry.output_height) {
        throw std::runtime_error(
            "BANet2D output spatial dimensions do not match input");
    }
    vision::Disparity result;
    result.map = std::make_shared<cv::Mat>(
        restore_banet_disparity(
            model_map,
            geometry,
            left.size()));
    const auto postprocess_end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_begin).count());
    set_runtime_total_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - total_begin).count());
    return result;
}

vision_core::InferResponse BANet2D::Run(
    const vision_core::InferRequest& request) {
    vision_core::InferResponse response;
    if (request.intent != vision_core::InferIntent::kStereoDepth) {
        response.ok = false;
        response.error_message =
            "BANet2D only supports kStereoDepth";
        return response;
    }
    const auto* stereo =
        std::get_if<vision_core::StereoImageInput>(&request.input);
    if (stereo == nullptr) {
        response.ok = false;
        response.error_message =
            "BANet2D expects StereoImageInput";
        return response;
    }
    response.results.emplace_back(infer_stereo(*stereo));
    return response;
}

std::vector<vision_core::InferIntent> BANet2D::supported_intents() const {
    return {vision_core::InferIntent::kStereoDepth};
}

std::vector<vision_core::ModelCapability> BANet2D::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<BANet2D> registrar("BANet2D");

}  // namespace vision_deploy
