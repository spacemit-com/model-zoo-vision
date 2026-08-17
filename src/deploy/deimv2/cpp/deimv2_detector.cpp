/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "deimv2_detector.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <sstream>
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
namespace {

constexpr int64_t kExpectedDetections = 300;
constexpr float kImageScale = 1.0F / 255.0F;
constexpr std::array<float, 3> kImageNetMean = {
    0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kImageNetStd = {
    0.229F, 0.224F, 0.225F};

vision_operators::ImagePreprocessSpec make_preprocess_spec(
    int width,
    int height,
    bool normalize) {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = width;
    spec.output_height = height;
    spec.resize_mode =
        vision_operators::PreprocessResizeMode::kLetterbox;
    spec.resize_rounding =
        vision_operators::PreprocessResizeRounding::kTruncate;
    spec.output_rgb = true;
    spec.padding = {0.0F, 0.0F, 0.0F};
    if (normalize) {
        for (size_t channel = 0; channel < 3; ++channel) {
            spec.mean[channel] = kImageNetMean[channel] * 255.0F;
            spec.scale[channel] =
                1.0F / (255.0F * kImageNetStd[channel]);
        }
    } else {
        spec.scale = {kImageScale, kImageScale, kImageScale};
    }
    return spec;
}

bool infer_normalize_from_model_path(const std::string& model_path) {
    std::string filename =
        std::filesystem::path(model_path).filename().string();
    std::transform(
        filename.begin(),
        filename.end(),
        filename.begin(),
        [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    if (filename.find("deimv2n") != std::string::npos) {
        return false;
    }
    if (filename.find("deimv2s") != std::string::npos ||
        filename.find("deimv2m") != std::string::npos) {
        return true;
    }
    throw std::runtime_error(
        "Cannot infer DEIMv2 preprocessing from model filename '" +
        filename + "'; set default_params.preprocess.normalize to "
        "true or false");
}

bool resolve_normalize(
    const YAML::Node& default_params,
    const std::string& model_path) {
    std::string mode = vision_core::yaml_utils::getString(
        default_params["preprocess"], "normalize", "auto");
    std::transform(
        mode.begin(),
        mode.end(),
        mode.begin(),
        [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    if (mode.empty() || mode == "auto") {
        return infer_normalize_from_model_path(model_path);
    }
    if (mode == "true") {
        return true;
    }
    if (mode == "false") {
        return false;
    }
    throw std::runtime_error(
        "default_params.preprocess.normalize must be auto, true, or false");
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

void require_shape(
    const std::vector<int64_t>& actual,
    const std::vector<int64_t>& expected,
    const std::string& tensor_name) {
    if (actual != expected) {
        throw std::runtime_error(
            "DEIMv2 tensor '" + tensor_name + "' has an unexpected shape");
    }
}

float score_at(const Ort::Value& scores, size_t index) {
    const auto type =
        scores.GetTensorTypeAndShapeInfo().GetElementType();
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        return static_cast<float>(
            scores.GetTensorData<Ort::Float16_t>()[index]);
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return scores.GetTensorData<float>()[index];
    }
    throw std::runtime_error(
        "DEIMv2 scores output must be float16 or float32");
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> DEIMv2Detector::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for DEIMv2Detector");
    }
    const YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error(
            "default_params not found in config for DEIMv2Detector");
    }
    const float conf_threshold = vision_core::yaml_utils::getFloat(
        default_params, "conf_threshold", 0.4F);
    const int num_threads = vision_core::yaml_utils::getInt(
        default_params, "num_threads", 8);
    const bool normalize = resolve_normalize(default_params, model_path);
    const std::string provider =
        vision_core::yaml_utils::getProvider(config);
    return std::make_unique<DEIMv2Detector>(
        model_path,
        conf_threshold,
        num_threads,
        normalize,
        lazy_load,
        provider);
}

DEIMv2Detector::DEIMv2Detector(
    const std::string& model_path,
    float conf_threshold,
    int num_threads,
    bool normalize,
    bool lazy_load,
    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        num_threads_(num_threads),
        normalize_(normalize),
        provider_(provider) {
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

void DEIMv2Detector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_names_.size() != 2 || input_names_[0] != "images" ||
        input_names_[1] != "orig_target_sizes") {
        throw std::runtime_error(
            "DEIMv2 expects inputs 'images' and 'orig_target_sizes'");
    }
    if (output_names_.size() != 3 || output_names_[0] != "labels" ||
        output_names_[1] != "boxes" || output_names_[2] != "scores") {
        throw std::runtime_error(
            "DEIMv2 expects outputs 'labels', 'boxes', and 'scores'");
    }

    const Ort::TypeInfo image_type_info =
        session_->GetInputTypeInfo(0);
    const auto image_info =
        image_type_info.GetTensorTypeAndShapeInfo();
    const auto image_shape = image_info.GetShape();
    if (image_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        image_shape.size() != 4 ||
        image_shape[0] != 1 || image_shape[1] != 3 ||
        image_shape[2] <= 0 || image_shape[3] <= 0) {
        throw std::runtime_error(
            "DEIMv2 images input must be float32 [1,3,H,W], got type " +
            std::to_string(static_cast<int>(image_info.GetElementType())) +
            " shape " +
            shape_string(image_shape));
    }
    const Ort::TypeInfo size_type_info =
        session_->GetInputTypeInfo(1);
    const auto size_info =
        size_type_info.GetTensorTypeAndShapeInfo();
    if (size_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::runtime_error(
            "DEIMv2 orig_target_sizes input must be int64");
    }
    require_shape(size_info.GetShape(), {1, 2}, "orig_target_sizes");

    const Ort::TypeInfo labels_type_info =
        session_->GetOutputTypeInfo(0);
    const Ort::TypeInfo boxes_type_info =
        session_->GetOutputTypeInfo(1);
    const Ort::TypeInfo scores_type_info =
        session_->GetOutputTypeInfo(2);
    const auto labels_info =
        labels_type_info.GetTensorTypeAndShapeInfo();
    const auto boxes_info =
        boxes_type_info.GetTensorTypeAndShapeInfo();
    const auto scores_info =
        scores_type_info.GetTensorTypeAndShapeInfo();
    if (labels_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
        boxes_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        (scores_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
        scores_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
        throw std::runtime_error("DEIMv2 output tensor types are unsupported");
    }
    require_shape(
        labels_info.GetShape(), {1, kExpectedDetections}, "labels");
    require_shape(
        boxes_info.GetShape(), {1, kExpectedDetections, 4}, "boxes");
    require_shape(
        scores_info.GetShape(), {1, kExpectedDetections}, "scores");
    model_loaded_ = true;
}

cv::Mat DEIMv2Detector::preprocess(const cv::Mat& image) const {
    const auto spec = make_preprocess_spec(
        static_cast<int>(input_shape_[3]),
        static_cast<int>(input_shape_[2]),
        normalize_);
    if (normalize_) {
        vision_operators::CpuChannelTransform transform;
        transform.input_divisor = {255.0F, 255.0F, 255.0F};
        transform.mean = kImageNetMean;
        transform.output_divisor = kImageNetStd;
        return vision_operators::preprocess_bgr_to_nchw(
            image, spec, transform);
    }
    return vision_operators::preprocess_bgr_to_nchw(image, spec);
}

DEIMv2Detector::LetterboxGeometry DEIMv2Detector::calculate_geometry(
    const cv::Size& original_size) const {
    if (original_size.width <= 0 || original_size.height <= 0) {
        throw std::runtime_error("DEIMv2 input image dimensions are invalid");
    }
    const int input_width = static_cast<int>(input_shape_[3]);
    const int input_height = static_cast<int>(input_shape_[2]);
    LetterboxGeometry geometry;
    geometry.ratio = std::min(
        static_cast<float>(input_width) / original_size.width,
        static_cast<float>(input_height) / original_size.height);
    const int resized_width =
        static_cast<int>(original_size.width * geometry.ratio);
    const int resized_height =
        static_cast<int>(original_size.height * geometry.ratio);
    geometry.left = (input_width - resized_width) / 2;
    geometry.top = (input_height - resized_height) / 2;
    return geometry;
}

std::vector<Ort::Value> DEIMv2Detector::run_session_two_inputs(
    const cv::Mat& image_tensor) {
    if (image_tensor.empty()) {
        throw std::runtime_error("DEIMv2 input tensor is empty");
    }
    std::vector<int64_t> image_shape = input_shape_;
    int64_t target_sizes[] = {
        image_shape[2], image_shape[3]};
    int64_t target_sizes_shape[] = {1, 2};

    std::vector<Ort::Value> inputs;
    inputs.reserve(2);
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(image_tensor.ptr<float>()),
        image_tensor.total(),
        image_shape.data(),
        image_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
        memory_info_,
        target_sizes,
        2,
        target_sizes_shape,
        2));

    return session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        inputs.data(),
        inputs.size(),
        output_node_names_.data(),
        output_node_names_.size());
}

vision_common::DetectionResultList DEIMv2Detector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& original_size,
    const LetterboxGeometry& geometry,
    float conf_threshold) const {
    if (outputs.size() != 3) {
        throw std::runtime_error("DEIMv2 inference must return three outputs");
    }
    const int64_t* labels = outputs[0].GetTensorData<int64_t>();
    const float* boxes = outputs[1].GetTensorData<float>();
    const float max_x = static_cast<float>(original_size.width);
    const float max_y = static_cast<float>(original_size.height);

    vision_common::DetectionResultList detections;
    detections.reserve(kExpectedDetections);
    for (size_t index = 0;
        index < static_cast<size_t>(kExpectedDetections);
        ++index) {
        const float score = score_at(outputs[2], index);
        if (score < conf_threshold) {
            continue;
        }
        const size_t offset = index * 4;
        const float x1 = std::clamp(
            (boxes[offset] - geometry.left) / geometry.ratio,
            0.0F,
            max_x);
        const float y1 = std::clamp(
            (boxes[offset + 1] - geometry.top) / geometry.ratio,
            0.0F,
            max_y);
        const float x2 = std::clamp(
            (boxes[offset + 2] - geometry.left) / geometry.ratio,
            0.0F,
            max_x);
        const float y2 = std::clamp(
            (boxes[offset + 3] - geometry.top) / geometry.ratio,
            0.0F,
            max_y);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        vision_common::DetectionResult detection;
        detection.bbox = {x1, y1, x2, y2};
        detection.score = score;
        detection.label = static_cast<int>(labels[index]);
        detections.push_back(detection);
    }
    return detections;
}

vision_common::DetectionResultList DEIMv2Detector::detect(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold) {
    (void)iou_threshold;
    vision_core::ImageInput input;
    input.image = image;
    return detect_input(input, conf_threshold);
}

vision_common::DetectionResultList DEIMv2Detector::detect_input(
    const vision_core::ImageInput& input,
    float conf_threshold) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto total_start = std::chrono::steady_clock::now();
    const float effective_confidence =
        conf_threshold > 0.0F ? conf_threshold : conf_threshold_;
    const cv::Size original_size(
        input.image.cols,
        input.format == vision_core::ImagePixelFormat::kNv12
            ? input.image.rows * 2 / 3
            : input.image.rows);
    const LetterboxGeometry geometry = calculate_geometry(original_size);

    const auto spec = make_preprocess_spec(
        static_cast<int>(input_shape_[3]),
        static_cast<int>(input_shape_[2]),
        normalize_);

    const auto preprocess_start = std::chrono::steady_clock::now();
    auto prepared = prepare_image(
        input,
        spec,
        [this](const cv::Mat& bgr) { return preprocess(bgr); });
    const auto preprocess_end = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(
        std::chrono::duration<double, std::milli>(
            preprocess_end - preprocess_start).count());

    const auto inference_start = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session_two_inputs(prepared.tensor());
    const auto inference_end = std::chrono::steady_clock::now();
    prepared.complete();
    set_runtime_model_infer_ms(
        std::chrono::duration<double, std::milli>(
            inference_end - inference_start).count());

    const auto postprocess_start = std::chrono::steady_clock::now();
    vision_common::DetectionResultList detections = postprocess(
        outputs, original_size, geometry, effective_confidence);
    const auto postprocess_end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_start).count());
    set_runtime_total_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - total_start).count());
    return detections;
}

vision_core::InferResponse DEIMv2Detector::Run(
    const vision_core::InferRequest& request) {
    assert(request.intent == vision_core::InferIntent::kDetect);
    const auto* image_input =
        std::get_if<vision_core::ImageInput>(&request.input);
    if (image_input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message = "DEIMv2Detector expects ImageInput";
        return response;
    }
    vision_common::DetectionResultList detections = detect_input(
        *image_input, request.params.conf_threshold);
    vision_core::InferResponse response;
    response.results.reserve(detections.size());
    for (auto& detection : detections) {
        response.results.emplace_back(std::move(detection));
    }
    return response;
}

std::vector<vision_core::InferIntent>
DEIMv2Detector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

std::vector<vision_core::ModelCapability>
DEIMv2Detector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<DEIMv2Detector> registrar(
    "DEIMv2Detector");

}  // namespace vision_deploy
