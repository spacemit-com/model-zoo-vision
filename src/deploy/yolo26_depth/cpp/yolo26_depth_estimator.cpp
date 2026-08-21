/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolo26_depth_estimator.h"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "operators/image_preprocess/image_preprocess_geometry.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {

double elapsed_ms(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(
        end - begin).count();
}

bool shape_is_positive(
    const std::vector<int64_t>& shape,
    size_t expected_rank) {
    if (shape.size() != expected_rank) {
        return false;
    }
    for (const int64_t dimension : shape) {
        if (dimension <= 0) {
            return false;
        }
    }
    return true;
}

std::string shape_string(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << '[';
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << shape[index];
    }
    stream << ']';
    return stream.str();
}

cv::Size validate_image_input(
    const vision_core::ImageInput& input) {
    if (input.image.empty()) {
        throw std::invalid_argument(
            "YOLO26-Depth input image is empty");
    }
    if (input.format == vision_core::ImagePixelFormat::kBgr8) {
        if (input.image.type() != CV_8UC3) {
            throw std::invalid_argument(
                "YOLO26-Depth BGR8 input must be CV_8UC3");
        }
        return input.image.size();
    }

    const int original_height = input.image.rows * 2 / 3;
    if (input.image.type() != CV_8UC1 ||
        input.image.rows % 3 != 0 ||
        (input.image.cols & 1) != 0 ||
        (original_height & 1) != 0) {
        throw std::invalid_argument(
            "YOLO26-Depth NV12 input must be CV_8UC1 "
            "H*3/2 x W with even H and W");
    }
    return cv::Size(input.image.cols, original_height);
}

vision_operators::ImagePreprocessSpec make_preprocess_spec(
    int input_width,
    int input_height) {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = input_width;
    spec.output_height = input_height;
    spec.resize_mode =
        vision_operators::PreprocessResizeMode::kLetterbox;
    spec.output_rgb = true;
    spec.interpolation =
        vision_operators::PreprocessInterpolation::kBilinear;
    spec.scale = {
        1.0F / 255.0F,
        1.0F / 255.0F,
        1.0F / 255.0F};
    spec.padding = {114.0F, 114.0F, 114.0F};
    return spec;
}

}  // namespace

YOLO26DepthEstimator::YOLO26DepthEstimator(
    const std::string& model_path,
    int num_threads,
    bool lazy_load,
    std::string provider)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        provider_(std::move(provider)) {
    if (num_threads_ <= 0) {
        throw std::invalid_argument(
            "YOLO26-Depth num_threads must be positive");
    }
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel>
YOLO26DepthEstimator::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(
            config,
            "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for "
            "YOLO26DepthEstimator");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<YOLO26DepthEstimator>(
        model_path,
        vision_core::yaml_utils::getInt(
            params,
            "num_threads",
            8),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void YOLO26DepthEstimator::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (session_->GetInputCount() != 1 ||
        session_->GetOutputCount() != 1) {
        throw std::runtime_error(
            "YOLO26-Depth expects one input and one output");
    }

    const Ort::TypeInfo input_type =
        session_->GetInputTypeInfo(0);
    const Ort::TypeInfo output_type =
        session_->GetOutputTypeInfo(0);
    const auto input_info =
        input_type.GetTensorTypeAndShapeInfo();
    const auto output_info =
        output_type.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> output_shape =
        output_info.GetShape();
    if (input_info.GetElementType() !=
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(
            "YOLO26-Depth input element type must be float32, got " +
            std::to_string(
                static_cast<int>(input_info.GetElementType())));
    }
    if (!shape_is_positive(input_shape_, 4) ||
        input_shape_[0] != 1 || input_shape_[1] != 3) {
        throw std::runtime_error(
            "YOLO26-Depth input shape must be [1,3,H,W], got " +
            shape_string(input_shape_));
    }
    if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        !shape_is_positive(output_shape, 4) ||
        output_shape[0] != 1 || output_shape[1] != 1 ||
        output_shape[2] != input_shape_[2] ||
        output_shape[3] != input_shape_[3]) {
        throw std::runtime_error(
            "YOLO26-Depth output must be float32 [1,1,H,W] "
            "with input spatial dimensions");
    }

    input_height_ = static_cast<int>(input_shape_[2]);
    input_width_ = static_cast<int>(input_shape_[3]);
    model_loaded_ = true;
}

cv::Mat YOLO26DepthEstimator::preprocess(
    const cv::Mat& bgr) const {
    if (bgr.empty() || bgr.type() != CV_8UC3 ||
        input_width_ <= 0 || input_height_ <= 0) {
        throw std::invalid_argument(
            "YOLO26-Depth expects a non-empty BGR8 image");
    }
    return vision_operators::preprocess_bgr_to_nchw(
        bgr,
        make_preprocess_spec(input_width_, input_height_));
}

cv::Mat YOLO26DepthEstimator::restore_depth(
    const Ort::Value& output,
    const cv::Size& original_size) const {
    if (!output.IsTensor() ||
        original_size.width <= 0 || original_size.height <= 0) {
        throw std::invalid_argument(
            "YOLO26-Depth output arguments are invalid");
    }
    const auto info = output.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    if (info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape != std::vector<int64_t>{
            1, 1, input_height_, input_width_}) {
        throw std::runtime_error(
            "YOLO26-Depth runtime output contract changed");
    }

    const vision_operators::ImagePreprocessGeometry geometry =
        vision_operators::make_image_preprocess_geometry(
            make_preprocess_spec(input_width_, input_height_),
            original_size.width,
            original_size.height);
    if (geometry.dst_x < 0 || geometry.dst_y < 0 ||
        geometry.dst_width <= 0 || geometry.dst_height <= 0 ||
        geometry.dst_x + geometry.dst_width > input_width_ ||
        geometry.dst_y + geometry.dst_height > input_height_) {
        throw std::runtime_error(
            "YOLO26-Depth letterbox geometry is invalid");
    }

    cv::Mat model_depth(
        input_height_,
        input_width_,
        CV_32FC1,
        const_cast<float*>(output.GetTensorData<float>()));
    const cv::Rect valid_region(
        geometry.dst_x,
        geometry.dst_y,
        geometry.dst_width,
        geometry.dst_height);
    cv::Mat restored;
    cv::resize(
        model_depth(valid_region),
        restored,
        original_size,
        0.0,
        0.0,
        cv::INTER_LINEAR);

    size_t valid_count = 0;
    for (int y = 0; y < restored.rows; ++y) {
        const float* row = restored.ptr<float>(y);
        for (int x = 0; x < restored.cols; ++x) {
            if (std::isfinite(row[x]) && row[x] > 0.0F) {
                ++valid_count;
            }
        }
    }
    if (valid_count == 0) {
        throw std::runtime_error(
            "YOLO26-Depth returned no positive finite depth values");
    }
    return restored;
}

vision::DepthMap YOLO26DepthEstimator::estimate_depth(
    const vision_core::ImageInput& input) {
    const cv::Size original_size = validate_image_input(input);
    ensure_model_loaded();
    reset_runtime_profile();

    const auto total_begin = std::chrono::steady_clock::now();
    const auto preprocess_begin = std::chrono::steady_clock::now();
    const vision_operators::ImagePreprocessSpec spec =
        make_preprocess_spec(input_width_, input_height_);
    auto prepared = prepare_image(
        input,
        spec,
        [this](const cv::Mat& bgr) {
            return preprocess(bgr);
        });
    const auto preprocess_end = std::chrono::steady_clock::now();

    const auto infer_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session(prepared.tensor());
    const auto infer_end = std::chrono::steady_clock::now();
    prepared.complete();
    if (outputs.size() != 1) {
        throw std::runtime_error(
            "YOLO26-Depth returned an invalid output");
    }

    const auto postprocess_begin = std::chrono::steady_clock::now();
    cv::Mat depth = restore_depth(outputs.front(), original_size);
    vision::DepthMap result;
    result.map = std::make_shared<cv::Mat>(std::move(depth));
    const auto postprocess_end = std::chrono::steady_clock::now();

    set_runtime_preprocess_ms(
        elapsed_ms(preprocess_begin, preprocess_end));
    set_runtime_model_infer_ms(
        elapsed_ms(infer_begin, infer_end));
    set_runtime_postprocess_ms(
        elapsed_ms(postprocess_begin, postprocess_end));
    set_runtime_total_ms(
        elapsed_ms(total_begin, postprocess_end));
    return result;
}

vision_core::InferResponse YOLO26DepthEstimator::Run(
    const vision_core::InferRequest& request) {
    if (request.intent !=
        vision_core::InferIntent::kMonocularDepth) {
        return unsupported_intent_response(request.intent);
    }
    const auto* input =
        std::get_if<vision_core::ImageInput>(&request.input);
    if (input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message =
            "YOLO26DepthEstimator expects ImageInput";
        return response;
    }

    vision_core::InferResponse response;
    response.results.emplace_back(estimate_depth(*input));
    return response;
}

std::vector<vision_core::InferIntent>
YOLO26DepthEstimator::supported_intents() const {
    return {vision_core::InferIntent::kMonocularDepth};
}

std::vector<vision_core::ModelCapability>
YOLO26DepthEstimator::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<YOLO26DepthEstimator>
registrar("YOLO26DepthEstimator");

}  // namespace vision_deploy
