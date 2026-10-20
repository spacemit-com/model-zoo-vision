/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rtdetr_detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "common/cpp/nms.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {
namespace {

cv::Size validate_image_input(const vision_core::ImageInput& input) {
    if (input.image.empty()) {
        throw std::invalid_argument("RT-DETR input image is empty");
    }
    if (input.format == vision_core::ImagePixelFormat::kBgr8) {
        if (input.image.type() != CV_8UC3) {
            throw std::invalid_argument(
                "RT-DETR BGR8 input must be CV_8UC3");
        }
        return input.image.size();
    }
    const int height = input.image.rows * 2 / 3;
    if (input.image.type() != CV_8UC1 ||
        input.image.rows % 3 != 0 ||
        (input.image.cols & 1) != 0 || (height & 1) != 0) {
        throw std::invalid_argument(
            "RT-DETR NV12 input must be CV_8UC1 H*3/2 x W "
            "with even H and W");
    }
    return cv::Size(input.image.cols, height);
}

vision_operators::ImagePreprocessSpec make_preprocess_spec(
    int width, int height) {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = width;
    spec.output_height = height;
    spec.resize_mode = vision_operators::PreprocessResizeMode::kStretch;
    spec.output_rgb = true;
    spec.scale = {1.0F / 255.0F, 1.0F / 255.0F, 1.0F / 255.0F};
    return spec;
}

}  // namespace

RTDETRDetector::RTDETRDetector(
    const std::string& model_path,
    float conf_threshold,
    int num_threads,
    bool lazy_load,
    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        num_threads_(num_threads),
        provider_(provider) {
    if (conf_threshold_ < 0.0F || conf_threshold_ > 1.0F ||
        num_threads_ <= 0) {
        throw std::invalid_argument(
            "RT-DETR conf_threshold must be in [0,1] and "
            "num_threads must be positive");
    }
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel> RTDETRDetector::create(
    const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(config, {"conf_threshold"});
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("RT-DETR model_path is missing");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<RTDETRDetector>(
        model_path,
        vision_core::yaml_utils::getFloat(
            params, "conf_threshold", 0.3F),
        vision_core::yaml_utils::getInt(params, "num_threads", 8),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void RTDETRDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_names_ !=
            std::vector<std::string>{"images", "orig_target_sizes"} ||
        output_names_ !=
            std::vector<std::string>{"labels", "boxes", "scores"}) {
        throw std::runtime_error("RT-DETR ONNX tensor names do not match");
    }
    const Ort::TypeInfo image_type = session_->GetInputTypeInfo(0);
    const auto image_info = image_type.GetTensorTypeAndShapeInfo();
    const auto image_shape = image_info.GetShape();
    if (image_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        image_shape.size() != 4 || image_shape[0] != 1 ||
        image_shape[1] != 3 || image_shape[2] <= 0 ||
        image_shape[3] <= 0) {
        std::ostringstream details;
        details << "RT-DETR images must be float32 [1,3,H,W], got type "
            << static_cast<int>(image_info.GetElementType()) << " shape [";
        for (size_t index = 0; index < image_shape.size(); ++index) {
            details << (index == 0 ? "" : ",") << image_shape[index];
        }
        details << "]";
        throw std::runtime_error(
            details.str());
    }
    const Ort::TypeInfo size_type = session_->GetInputTypeInfo(1);
    const auto size_info = size_type.GetTensorTypeAndShapeInfo();
    if (size_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        size_info.GetShape() != std::vector<int64_t>{1, 2}) {
        throw std::runtime_error(
            "RT-DETR orig_target_sizes must be int64 [1,2]");
    }
    const Ort::TypeInfo labels_type = session_->GetOutputTypeInfo(0);
    const Ort::TypeInfo boxes_type = session_->GetOutputTypeInfo(1);
    const Ort::TypeInfo scores_type = session_->GetOutputTypeInfo(2);
    const auto labels_info = labels_type.GetTensorTypeAndShapeInfo();
    const auto boxes_info = boxes_type.GetTensorTypeAndShapeInfo();
    const auto scores_info = scores_type.GetTensorTypeAndShapeInfo();
    const auto labels_shape = labels_info.GetShape();
    if (labels_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        boxes_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        scores_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        labels_shape.size() != 2 || labels_shape[0] != 1 ||
        labels_shape[1] <= 0 ||
        boxes_info.GetShape() !=
            std::vector<int64_t>{1, labels_shape[1], 4} ||
        scores_info.GetShape() != labels_shape) {
        throw std::runtime_error(
            "RT-DETR outputs must be labels [1,N], "
            "boxes [1,N,4], scores [1,N]");
    }
    input_height_ = static_cast<int>(input_shape_[2]);
    input_width_ = static_cast<int>(input_shape_[3]);
    num_detections_ = labels_shape[1];
    model_loaded_ = true;
}

cv::Mat RTDETRDetector::preprocess(const cv::Mat& bgr) const {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument(
            "RT-DETR expects a non-empty BGR8 image");
    }
    return vision_operators::preprocess_bgr_to_nchw(
        bgr, make_preprocess_spec(input_width_, input_height_));
}

vision_common::DetectionResultList RTDETRDetector::postprocess(
    const std::vector<Ort::Value>& outputs,
    const cv::Size& original_size,
    float conf_threshold) const {
    if (outputs.size() != 3 ||
        outputs[0].GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>{1, num_detections_} ||
        outputs[1].GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>{1, num_detections_, 4} ||
        outputs[2].GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>{1, num_detections_}) {
        throw std::runtime_error("RT-DETR runtime output shape changed");
    }
    const int64_t* labels = outputs[0].GetTensorData<int64_t>();
    const float* boxes = outputs[1].GetTensorData<float>();
    const float* scores = outputs[2].GetTensorData<float>();
    vision_common::DetectionResultList detections;
    detections.reserve(static_cast<size_t>(num_detections_));
    for (int64_t index = 0; index < num_detections_; ++index) {
        if (!std::isfinite(scores[index]) ||
            scores[index] < conf_threshold ||
            labels[index] < 0 || labels[index] >= 80) {
            continue;
        }
        const size_t offset = static_cast<size_t>(index) * 4;
        if (!std::isfinite(boxes[offset]) ||
            !std::isfinite(boxes[offset + 1]) ||
            !std::isfinite(boxes[offset + 2]) ||
            !std::isfinite(boxes[offset + 3])) {
            continue;
        }
        // The exported postprocessor has already scaled xyxy to the
        // supplied original width and height; no NMS or inverse resize.
        const float x1 = std::clamp(
            boxes[offset], 0.0F,
            static_cast<float>(original_size.width));
        const float y1 = std::clamp(
            boxes[offset + 1], 0.0F,
            static_cast<float>(original_size.height));
        const float x2 = std::clamp(
            boxes[offset + 2], 0.0F,
            static_cast<float>(original_size.width));
        const float y2 = std::clamp(
            boxes[offset + 3], 0.0F,
            static_cast<float>(original_size.height));
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        vision_common::DetectionResult detection;
        detection.bbox = {x1, y1, x2, y2};
        detection.score = scores[index];
        detection.label = static_cast<int>(labels[index]);
        detections.push_back(detection);
    }
    return detections;
}

vision_common::DetectionResultList RTDETRDetector::infer_input(
    const vision_core::ImageInput& input,
    float conf_threshold, int max_det) {
    ensure_model_loaded();
    reset_runtime_profile();
    const cv::Size original_size = validate_image_input(input);
    const auto total_start = std::chrono::steady_clock::now();

    const auto preprocess_start = std::chrono::steady_clock::now();
    auto prepared = prepare_image(
        input, make_preprocess_spec(input_width_, input_height_),
        [this](const cv::Mat& bgr) { return preprocess(bgr); });
    const auto preprocess_end = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(
        std::chrono::duration<double, std::milli>(
            preprocess_end - preprocess_start).count());

    int64_t target_sizes[] = {
        static_cast<int64_t>(original_size.width),
        static_cast<int64_t>(original_size.height)};
    int64_t target_shape[] = {1, 2};
    std::vector<Ort::Value> inputs;
    inputs.reserve(2);
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(prepared.tensor().ptr<float>()),
        prepared.tensor().total(),
        input_shape_.data(), input_shape_.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_, target_sizes, 2, target_shape, 2));
    const auto inference_start = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_node_names_.data(),
        inputs.data(), inputs.size(), output_node_names_.data(),
        output_node_names_.size());
    const auto inference_end = std::chrono::steady_clock::now();
    prepared.complete();
    set_runtime_model_infer_ms(
        std::chrono::duration<double, std::milli>(
            inference_end - inference_start).count());

    const auto postprocess_start = std::chrono::steady_clock::now();
    auto detections = postprocess(outputs, original_size, conf_threshold);
    vision_common::filter_and_limit_detections(detections, max_det);
    const auto postprocess_end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_start).count());
    set_runtime_total_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - total_start).count());
    return detections;
}

vision_core::InferResponse RTDETRDetector::Run(
    const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kDetect) {
        return unsupported_intent_response(request.intent);
    }
    const auto* image_input =
        std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "RTDETRDetector expects ImageInput";
        return response;
    }
    const float threshold = vision_core::resolve_probability_threshold(
        request.params.conf_threshold, conf_threshold_, "conf_threshold");
    auto detections = infer_input(*image_input, threshold, request.params.max_det);
    vision_core::InferResponse response;
    response.results.reserve(detections.size());
    for (auto& detection : detections) {
        response.results.emplace_back(std::move(detection));
    }
    return response;
}

std::vector<vision_core::InferIntent>
RTDETRDetector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

std::vector<vision_core::ModelCapability>
RTDETRDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<RTDETRDetector> registrar(
    "RTDETRDetector");

}  // namespace vision_deploy
