/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rfdetr_detector.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {
namespace {

constexpr int64_t kCocoSparseClassCount = 91;
constexpr std::array<float, 3> kImageNetMean = {
    0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kImageNetStd = {
    0.229F, 0.224F, 0.225F};

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

cv::Size validate_image_input(const vision_core::ImageInput& input) {
    if (input.image.empty()) {
        throw std::invalid_argument("RF-DETR input image is empty");
    }
    if (input.format == vision_core::ImagePixelFormat::kBgr8) {
        if (input.image.type() != CV_8UC3) {
            throw std::invalid_argument(
                "RF-DETR BGR8 input must be CV_8UC3");
        }
        return input.image.size();
    }

    const int original_height = input.image.rows * 2 / 3;
    if (input.image.type() != CV_8UC1 ||
        input.image.rows % 3 != 0 ||
        (input.image.cols & 1) != 0 ||
        (original_height & 1) != 0) {
        throw std::invalid_argument(
            "RF-DETR NV12 input must be CV_8UC1 H*3/2 x W "
            "with even H and W");
    }
    return cv::Size(input.image.cols, original_height);
}

vision_operators::ImagePreprocessSpec make_preprocess_spec(
    int input_width,
    int input_height) {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = input_width;
    spec.output_height = input_height;
    spec.resize_mode = vision_operators::PreprocessResizeMode::kStretch;
    spec.output_rgb = true;
    spec.interpolation =
        vision_operators::PreprocessInterpolation::kBilinear;
    for (size_t channel = 0; channel < 3; ++channel) {
        spec.mean[channel] = kImageNetMean[channel] * 255.0F;
        spec.scale[channel] =
            1.0F / (255.0F * kImageNetStd[channel]);
    }
    return spec;
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-value));
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

struct Candidate {
    size_t flat_index = 0;
    float score = 0.0F;
};

bool candidate_greater(const Candidate& left, const Candidate& right) {
    if (left.score != right.score) {
        return left.score > right.score;
    }
    return left.flat_index < right.flat_index;
}

}  // namespace

RFDETRDetector::RFDETRDetector(
    const std::string& model_path,
    float conf_threshold,
    int max_det,
    int num_threads,
    bool lazy_load,
    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        max_det_(max_det),
        num_threads_(num_threads),
        provider_(provider) {
    if (conf_threshold_ < 0.0F || conf_threshold_ > 1.0F) {
        throw std::invalid_argument(
            "RF-DETR conf_threshold must be in [0,1]");
    }
    if (max_det_ <= 0 || num_threads_ <= 0) {
        throw std::invalid_argument(
            "RF-DETR max_det and num_threads must be positive");
    }
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel> RFDETRDetector::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for RFDETRDetector");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<RFDETRDetector>(
        model_path,
        vision_core::yaml_utils::getFloat(
            params, "conf_threshold", 0.3F),
        vision_core::yaml_utils::getInt(params, "max_det", 300),
        vision_core::yaml_utils::getInt(params, "num_threads", 8),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void RFDETRDetector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_names_.size() != 1 || input_names_[0] != "input") {
        throw std::runtime_error("RF-DETR expects one input named 'input'");
    }
    if (output_names_.size() != 2 || output_names_[0] != "dets" ||
        output_names_[1] != "labels") {
        throw std::runtime_error(
            "RF-DETR expects outputs 'dets' and 'labels'");
    }

    const Ort::TypeInfo input_type = session_->GetInputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        input_shape_.size() != 4 || input_shape_[0] != 1 ||
        input_shape_[1] != 3 || input_shape_[2] <= 0 ||
        input_shape_[3] <= 0) {
        throw std::runtime_error(
            "RF-DETR input must be float32 [1,3,H,W], got " +
            shape_string(input_shape_));
    }

    const Ort::TypeInfo boxes_type = session_->GetOutputTypeInfo(0);
    const Ort::TypeInfo logits_type = session_->GetOutputTypeInfo(1);
    const auto boxes_info = boxes_type.GetTensorTypeAndShapeInfo();
    const auto logits_info = logits_type.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> boxes_shape = boxes_info.GetShape();
    const std::vector<int64_t> logits_shape = logits_info.GetShape();
    if (boxes_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        boxes_shape.size() != 3 || boxes_shape[0] != 1 ||
        boxes_shape[1] <= 0 || boxes_shape[2] != 4) {
        throw std::runtime_error(
            "RF-DETR dets output must be float32 [1,Q,4], got " +
            shape_string(boxes_shape));
    }
    if (logits_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        logits_shape.size() != 3 || logits_shape[0] != 1 ||
        logits_shape[1] != boxes_shape[1] ||
        logits_shape[2] != kCocoSparseClassCount) {
        throw std::runtime_error(
            "RF-DETR labels output must be float32 [1,Q,91], got " +
            shape_string(logits_shape));
    }

    input_height_ = static_cast<int>(input_shape_[2]);
    input_width_ = static_cast<int>(input_shape_[3]);
    num_queries_ = boxes_shape[1];
    num_classes_ = logits_shape[2];
    model_loaded_ = true;
}

cv::Mat RFDETRDetector::preprocess(const cv::Mat& bgr) const {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument(
            "RF-DETR expects a non-empty BGR8 image");
    }
    vision_operators::CpuChannelTransform transform;
    transform.input_divisor = {255.0F, 255.0F, 255.0F};
    transform.mean = kImageNetMean;
    transform.output_divisor = kImageNetStd;
    return vision_operators::preprocess_bgr_to_nchw(
        bgr,
        make_preprocess_spec(input_width_, input_height_),
        transform);
}

vision_common::DetectionResultList RFDETRDetector::postprocess(
    const std::vector<Ort::Value>& outputs,
    const cv::Size& original_size,
    float conf_threshold,
    int max_det) const {
    if (outputs.size() != 2 || !outputs[0].IsTensor() ||
        !outputs[1].IsTensor()) {
        throw std::runtime_error(
            "RF-DETR inference must return two tensors");
    }
    const auto boxes_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto logits_info = outputs[1].GetTensorTypeAndShapeInfo();
    if (boxes_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        logits_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        boxes_info.GetShape() !=
            std::vector<int64_t>{1, num_queries_, 4} ||
        logits_info.GetShape() !=
            std::vector<int64_t>{1, num_queries_, num_classes_}) {
        throw std::runtime_error(
            "RF-DETR runtime output contract changed");
    }

    const float* boxes = outputs[0].GetTensorData<float>();
    const float* logits = outputs[1].GetTensorData<float>();
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(num_queries_));
    const size_t value_count = static_cast<size_t>(
        num_queries_ * num_classes_);
    for (size_t index = 0; index < value_count; ++index) {
        if (!std::isfinite(logits[index])) {
            continue;
        }
        const float score = sigmoid(logits[index]);
        if (score > conf_threshold) {
            candidates.push_back({index, score});
        }
    }

    const size_t limit = std::min({
        candidates.size(),
        static_cast<size_t>(max_det),
        static_cast<size_t>(num_queries_)});
    if (limit < candidates.size()) {
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(limit),
            candidates.end(),
            candidate_greater);
        candidates.resize(limit);
    } else {
        std::sort(
            candidates.begin(), candidates.end(), candidate_greater);
    }

    vision_common::DetectionResultList detections;
    detections.reserve(candidates.size());
    const float image_width = static_cast<float>(original_size.width);
    const float image_height = static_cast<float>(original_size.height);
    for (const Candidate& candidate : candidates) {
        const size_t query =
            candidate.flat_index / static_cast<size_t>(num_classes_);
        const int label = static_cast<int>(
            candidate.flat_index % static_cast<size_t>(num_classes_));
        const size_t offset = query * 4;
        const float center_x = boxes[offset];
        const float center_y = boxes[offset + 1];
        const float width = boxes[offset + 2];
        const float height = boxes[offset + 3];
        if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
            !std::isfinite(width) || !std::isfinite(height)) {
            continue;
        }
        const float x1 = std::clamp(
            (center_x - width * 0.5F) * image_width,
            0.0F,
            image_width);
        const float y1 = std::clamp(
            (center_y - height * 0.5F) * image_height,
            0.0F,
            image_height);
        const float x2 = std::clamp(
            (center_x + width * 0.5F) * image_width,
            0.0F,
            image_width);
        const float y2 = std::clamp(
            (center_y + height * 0.5F) * image_height,
            0.0F,
            image_height);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        vision_common::DetectionResult detection;
        detection.bbox = {x1, y1, x2, y2};
        detection.score = candidate.score;
        detection.label = label;
        detections.push_back(detection);
    }
    return detections;
}

vision_common::DetectionResultList RFDETRDetector::infer_input(
    const vision_core::ImageInput& input,
    float conf_threshold,
    int max_det) {
    ensure_model_loaded();
    reset_runtime_profile();
    const cv::Size original_size = validate_image_input(input);
    const auto total_start = std::chrono::steady_clock::now();

    const auto preprocess_start = std::chrono::steady_clock::now();
    auto prepared = prepare_image(
        input,
        make_preprocess_spec(input_width_, input_height_),
        [this](const cv::Mat& bgr) { return preprocess(bgr); });
    const auto preprocess_end = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(
        std::chrono::duration<double, std::milli>(
            preprocess_end - preprocess_start).count());

    std::vector<Ort::Value> outputs = run_session(prepared.tensor());
    prepared.complete();

    const auto postprocess_start = std::chrono::steady_clock::now();
    vision_common::DetectionResultList detections = postprocess(
        outputs, original_size, conf_threshold, max_det);
    const auto postprocess_end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_start).count());
    set_runtime_total_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - total_start).count());
    return detections;
}

vision_core::InferResponse RFDETRDetector::Run(
    const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kDetect) {
        return unsupported_intent_response(request.intent);
    }
    const auto* image_input =
        std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "RFDETRDetector expects ImageInput";
        return response;
    }
    const float confidence = request.params.conf_threshold > 0.0F
        ? request.params.conf_threshold : conf_threshold_;
    const int max_det = request.params.max_det > 0
        ? request.params.max_det : max_det_;
    vision_common::DetectionResultList detections = infer_input(
        *image_input, confidence, max_det);
    vision_core::InferResponse response;
    response.results.reserve(detections.size());
    for (auto& detection : detections) {
        response.results.emplace_back(std::move(detection));
    }
    return response;
}

std::vector<vision_core::InferIntent>
RFDETRDetector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

std::vector<vision_core::ModelCapability>
RFDETRDetector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<RFDETRDetector> registrar(
    "RFDETRDetector");

}  // namespace vision_deploy
