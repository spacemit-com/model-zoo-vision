/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolo26_semantic_segmentor.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
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

cv::Size validate_image_input(
    const vision_core::ImageInput& input) {
    if (input.image.empty()) {
        throw std::invalid_argument(
            "YOLO26-Sem input image is empty");
    }
    if (input.format == vision_core::ImagePixelFormat::kBgr8) {
        if (input.image.type() != CV_8UC3) {
            throw std::invalid_argument(
                "YOLO26-Sem BGR8 input must be CV_8UC3");
        }
        return input.image.size();
    }

    const int original_height = input.image.rows * 2 / 3;
    if (input.image.type() != CV_8UC1 ||
        input.image.rows % 3 != 0 ||
        (input.image.cols & 1) != 0 ||
        (original_height & 1) != 0) {
        throw std::invalid_argument(
            "YOLO26-Sem NV12 input must be CV_8UC1 "
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

YOLO26SemanticSegmentor::YOLO26SemanticSegmentor(
    const std::string& model_path,
    int num_threads,
    int num_classes,
    bool lazy_load,
    std::string provider)
    : BaseModel(model_path, lazy_load),
        num_threads_(num_threads),
        num_classes_(num_classes),
        provider_(std::move(provider)) {
    if (num_threads_ <= 0 ||
        num_classes_ <= 0 || num_classes_ > 256) {
        throw std::invalid_argument(
            "YOLO26-Sem parameters are invalid");
    }
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel>
YOLO26SemanticSegmentor::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(
            config,
            "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for "
            "YOLO26SemanticSegmentor");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<YOLO26SemanticSegmentor>(
        model_path,
        vision_core::yaml_utils::getInt(
            params,
            "num_threads",
            8),
        vision_core::yaml_utils::getInt(
            params,
            "num_classes",
            19),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void YOLO26SemanticSegmentor::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (session_->GetInputCount() != 1 ||
        session_->GetOutputCount() != 1) {
        throw std::runtime_error(
            "YOLO26-Sem expects one input and one output");
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
            "YOLO26-Sem input element type must be float32, got " +
            std::to_string(
                static_cast<int>(input_info.GetElementType())));
    }
    if (!shape_is_positive(input_shape_, 4) ||
        input_shape_[0] != 1 || input_shape_[1] != 3) {
        throw std::runtime_error(
            "YOLO26-Sem input shape must be [1,3,H,W]");
    }
    if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        !shape_is_positive(output_shape, 4) ||
        output_shape[0] != 1 ||
        output_shape[1] != num_classes_ ||
        output_shape[2] != input_shape_[2] ||
        output_shape[3] != input_shape_[3]) {
        throw std::runtime_error(
            "YOLO26-Sem output must be float32 [1,C,H,W] "
            "with input spatial dimensions");
    }

    input_height_ = static_cast<int>(input_shape_[2]);
    input_width_ = static_cast<int>(input_shape_[3]);
    model_loaded_ = true;
}

cv::Mat YOLO26SemanticSegmentor::preprocess(
    const cv::Mat& bgr) const {
    if (bgr.empty() || bgr.type() != CV_8UC3 ||
        input_width_ <= 0 || input_height_ <= 0) {
        throw std::invalid_argument(
            "YOLO26-Sem expects a non-empty BGR8 image");
    }
    return vision_operators::preprocess_bgr_to_nchw(
        bgr,
        make_preprocess_spec(input_width_, input_height_));
}

cv::Mat YOLO26SemanticSegmentor::decode_label_map(
    const Ort::Value& output,
    const cv::Size& original_size) const {
    if (!output.IsTensor() ||
        original_size.width <= 0 || original_size.height <= 0) {
        throw std::invalid_argument(
            "YOLO26-Sem output arguments are invalid");
    }
    const auto info = output.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    if (info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape != std::vector<int64_t>{
            1, num_classes_, input_height_, input_width_}) {
        throw std::runtime_error(
            "YOLO26-Sem runtime output contract changed");
    }

    const vision_operators::ImagePreprocessGeometry geometry =
        vision_operators::make_image_preprocess_geometry(
            make_preprocess_spec(input_width_, input_height_),
            original_size.width,
            original_size.height);
    if (geometry.dst_x < 0 || geometry.dst_y < 0 ||
        geometry.dst_x + geometry.dst_width > input_width_ ||
        geometry.dst_y + geometry.dst_height > input_height_) {
        throw std::runtime_error(
            "YOLO26-Sem letterbox geometry is invalid");
    }

    const cv::Rect valid_region(
        geometry.dst_x,
        geometry.dst_y,
        geometry.dst_width,
        geometry.dst_height);
    const size_t model_area =
        static_cast<size_t>(input_height_) * input_width_;
    const float* logits = output.GetTensorData<float>();

    cv::Mat best_scores(
        original_size,
        CV_32FC1,
        cv::Scalar(-std::numeric_limits<float>::infinity()));
    cv::Mat label_map(
        original_size,
        CV_8UC1,
        cv::Scalar(0));
    cv::Mat restored;
    cv::Mat update_mask;
    for (int class_id = 0;
        class_id < num_classes_;
        ++class_id) {
        cv::Mat model_logits(
            input_height_,
            input_width_,
            CV_32FC1,
            const_cast<float*>(
                logits + static_cast<size_t>(class_id) * model_area));
        cv::resize(
            model_logits(valid_region),
            restored,
            original_size,
            0.0,
            0.0,
            cv::INTER_LINEAR);
        if (class_id == 0) {
            restored.copyTo(best_scores);
            continue;
        }
        cv::compare(
            restored,
            best_scores,
            update_mask,
            cv::CMP_GT);
        restored.copyTo(best_scores, update_mask);
        label_map.setTo(class_id, update_mask);
    }
    return label_map;
}

std::vector<vision::Segmentation>
YOLO26SemanticSegmentor::split_semantic_masks(
    const cv::Mat& label_map) const {
    if (label_map.empty() || label_map.type() != CV_8UC1) {
        throw std::invalid_argument(
            "YOLO26-Sem label map must be CV_8UC1");
    }

    std::vector<vision::Segmentation> results;
    for (int class_id = 0;
        class_id < num_classes_;
        ++class_id) {
        cv::Mat mask;
        cv::compare(label_map, class_id, mask, cv::CMP_EQ);
        if (cv::countNonZero(mask) == 0) {
            continue;
        }
        vision::Segmentation result;
        result.bbox = {-1.0F, -1.0F, -1.0F, -1.0F};
        result.score = 1.0F;
        result.label = class_id;
        result.mask = std::make_shared<cv::Mat>(
            std::move(mask));
        results.push_back(std::move(result));
    }
    return results;
}

std::vector<vision::Segmentation>
YOLO26SemanticSegmentor::segment(
    const vision_core::ImageInput& input) {
    const cv::Size original_size = validate_image_input(input);
    ensure_model_loaded();
    reset_runtime_profile();

    const auto total_begin =
        std::chrono::steady_clock::now();
    const auto preprocess_begin =
        std::chrono::steady_clock::now();
    const vision_operators::ImagePreprocessSpec spec =
        make_preprocess_spec(input_width_, input_height_);
    auto prepared = prepare_image(
        input,
        spec,
        [this](const cv::Mat& bgr) {
            return preprocess(bgr);
        });
    const auto preprocess_end =
        std::chrono::steady_clock::now();

    const auto infer_begin =
        std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session(prepared.tensor());
    const auto infer_end =
        std::chrono::steady_clock::now();
    prepared.complete();
    if (outputs.size() != 1) {
        throw std::runtime_error(
            "YOLO26-Sem returned an invalid output");
    }

    const auto postprocess_begin =
        std::chrono::steady_clock::now();
    const cv::Mat label_map =
        decode_label_map(outputs[0], original_size);
    std::vector<vision::Segmentation> results =
        split_semantic_masks(label_map);
    const auto postprocess_end =
        std::chrono::steady_clock::now();

    set_runtime_preprocess_ms(
        elapsed_ms(preprocess_begin, preprocess_end));
    set_runtime_model_infer_ms(
        elapsed_ms(infer_begin, infer_end));
    set_runtime_postprocess_ms(
        elapsed_ms(postprocess_begin, postprocess_end));
    set_runtime_total_ms(
        elapsed_ms(total_begin, postprocess_end));
    return results;
}

vision_core::InferResponse YOLO26SemanticSegmentor::Run(
    const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kSegment) {
        return unsupported_intent_response(request.intent);
    }
    const auto* input =
        std::get_if<vision_core::ImageInput>(&request.input);
    if (input == nullptr) {
        vision_core::InferResponse response;
        response.ok = false;
        response.error_message =
            "YOLO26SemanticSegmentor expects ImageInput";
        return response;
    }

    std::vector<vision::Segmentation> results = segment(*input);
    vision_core::InferResponse response;
    response.results.reserve(results.size());
    for (auto& result : results) {
        response.results.emplace_back(std::move(result));
    }
    return response;
}

std::vector<vision_core::InferIntent>
YOLO26SemanticSegmentor::supported_intents() const {
    return {vision_core::InferIntent::kSegment};
}

std::vector<vision_core::ModelCapability>
YOLO26SemanticSegmentor::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<YOLO26SemanticSegmentor>
registrar("YOLO26SemanticSegmentor");

}  // namespace vision_deploy
