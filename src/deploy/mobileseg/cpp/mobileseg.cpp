/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mobileseg.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "operators/image_preprocess/cpu_image_preprocessor.h"
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

vision_operators::ImagePreprocessSpec make_mobileseg_preprocess_spec(
    int input_width,
    int input_height)
{
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = input_width;
    spec.output_height = input_height;
    spec.resize_mode =
        vision_operators::PreprocessResizeMode::kStretch;
    spec.output_rgb = true;
    spec.interpolation =
        vision_operators::PreprocessInterpolation::kBilinear;
    spec.mean = {127.5F, 127.5F, 127.5F};
    spec.scale = {
        1.0F / 127.5F,
        1.0F / 127.5F,
        1.0F / 127.5F};
    return spec;
}

}  // namespace

cv::Size validate_mobileseg_image_input(
    const vision_core::ImageInput& input) {
    if (input.image.empty()) {
        throw std::invalid_argument(
            "MobileSeg input image is empty");
    }
    if (input.format ==
        vision_core::ImagePixelFormat::kBgr8) {
        if (input.image.type() != CV_8UC3) {
            throw std::invalid_argument(
                "MobileSeg BGR8 input must be CV_8UC3");
        }
        return input.image.size();
    }

    const int original_height =
        input.image.rows * 2 / 3;
    if (input.image.type() != CV_8UC1 ||
        input.image.rows % 3 != 0 ||
        (input.image.cols & 1) != 0 ||
        (original_height & 1) != 0) {
        throw std::invalid_argument(
            "MobileSeg NV12 input must be CV_8UC1 "
            "H*3/2 x W with even H and W");
    }
    return cv::Size(
        input.image.cols,
        original_height);
}

cv::Mat decode_mobileseg_label_map(
    const int32_t* labels,
    int model_height,
    int model_width,
    const cv::Size& original_size,
    int num_classes) {
    if (labels == nullptr ||
        model_height <= 0 || model_width <= 0 ||
        original_size.width <= 0 || original_size.height <= 0 ||
        num_classes <= 0 || num_classes > 256) {
        throw std::invalid_argument(
            "MobileSeg label-map arguments are invalid");
    }

    cv::Mat model_labels(model_height, model_width, CV_8UC1);
    for (int y = 0; y < model_height; ++y) {
        uint8_t* destination = model_labels.ptr<uint8_t>(y);
        for (int x = 0; x < model_width; ++x) {
            const int32_t label =
                labels[static_cast<size_t>(y) * model_width + x];
            if (label < 0 || label >= num_classes) {
                throw std::runtime_error(
                    "MobileSeg returned an out-of-range class id");
            }
            destination[x] = static_cast<uint8_t>(label);
        }
    }

    if (model_labels.size() == original_size) {
        return model_labels;
    }
    cv::Mat restored;
    cv::resize(
        model_labels,
        restored,
        original_size,
        0.0,
        0.0,
        cv::INTER_NEAREST);
    return restored;
}

std::vector<vision::Segmentation>
split_mobileseg_semantic_masks(
    const cv::Mat& label_map,
    int num_classes) {
    if (label_map.empty() ||
        label_map.type() != CV_8UC1 ||
        num_classes <= 0 || num_classes > 256) {
        throw std::invalid_argument(
            "MobileSeg semantic-mask arguments are invalid");
    }

    std::vector<vision::Segmentation> results;
    for (int class_id = 0;
        class_id < num_classes;
        ++class_id) {
        cv::Mat mask;
        cv::compare(
            label_map,
            class_id,
            mask,
            cv::CMP_EQ);
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

MobileSeg::MobileSeg(
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
            "MobileSeg parameters are invalid");
    }
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel> MobileSeg::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(
            config,
            "model_path");
    if (model_path.empty()) {
        throw std::runtime_error(
            "model_path not found in config for MobileSeg");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<MobileSeg>(
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

void MobileSeg::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (session_->GetInputCount() != 1 ||
        session_->GetOutputCount() != 1) {
        throw std::runtime_error(
            "MobileSeg expects one input and one output");
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
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        !shape_is_positive(input_shape_, 4) ||
        input_shape_[0] != 1 ||
        input_shape_[1] != 3) {
        throw std::runtime_error(
            "MobileSeg input must be float32 [1,3,H,W]");
    }
    if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
        !shape_is_positive(output_shape, 3) ||
        output_shape[0] != 1 ||
        output_shape[1] != input_shape_[2] ||
        output_shape[2] != input_shape_[3]) {
        throw std::runtime_error(
            "MobileSeg output must be int32 [1,H,W] "
            "with input spatial dimensions");
    }

    input_height_ = static_cast<int>(input_shape_[2]);
    input_width_ = static_cast<int>(input_shape_[3]);
    model_loaded_ = true;
}

cv::Mat MobileSeg::preprocess(const cv::Mat& bgr) const {
    if (bgr.empty() || bgr.type() != CV_8UC3 ||
        input_width_ <= 0 || input_height_ <= 0) {
        throw std::invalid_argument(
            "MobileSeg expects a non-empty BGR8 image");
    }

    return vision_operators::preprocess_bgr_to_nchw(
        bgr,
        make_mobileseg_preprocess_spec(
            input_width_, input_height_));
}

std::vector<vision::Segmentation> MobileSeg::segment(
    const vision_core::ImageInput& input) {
    const cv::Size original_size =
        validate_mobileseg_image_input(input);
    ensure_model_loaded();

    reset_runtime_profile();
    const auto total_begin =
        std::chrono::steady_clock::now();
    const auto preprocess_begin =
        std::chrono::steady_clock::now();
    const vision_operators::ImagePreprocessSpec spec =
        make_mobileseg_preprocess_spec(
            input_width_, input_height_);
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
    if (outputs.size() != 1 || !outputs[0].IsTensor()) {
        throw std::runtime_error(
            "MobileSeg returned an invalid output");
    }
    const auto output_info =
        outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> output_shape =
        output_info.GetShape();
    if (output_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
        output_shape !=
            std::vector<int64_t>{
                1, input_height_, input_width_}) {
        throw std::runtime_error(
            "MobileSeg runtime output contract changed");
    }

    const auto postprocess_begin =
        std::chrono::steady_clock::now();
    const cv::Mat label_map =
        decode_mobileseg_label_map(
            outputs[0].GetTensorData<int32_t>(),
            input_height_,
            input_width_,
            original_size,
            num_classes_);
    std::vector<vision::Segmentation> results =
        split_mobileseg_semantic_masks(
            label_map,
            num_classes_);
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

vision_core::InferResponse MobileSeg::Run(
    const vision_core::InferRequest& request) {
    vision_core::InferResponse response;
    if (request.intent !=
        vision_core::InferIntent::kSegment) {
        response.ok = false;
        response.error_message =
            "MobileSeg only supports kSegment";
        return response;
    }
    const auto* input =
        std::get_if<vision_core::ImageInput>(
            &request.input);
    if (input == nullptr) {
        response.ok = false;
        response.error_message =
            "MobileSeg expects ImageInput";
        return response;
    }

    std::vector<vision::Segmentation> results =
        segment(*input);
    response.results.reserve(results.size());
    for (auto& result : results) {
        response.results.emplace_back(
            std::move(result));
    }
    return response;
}

std::vector<vision_core::InferIntent>
MobileSeg::supported_intents() const {
    return {vision_core::InferIntent::kSegment};
}

std::vector<vision_core::ModelCapability>
MobileSeg::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<MobileSeg> registrar(
    "MobileSeg");

}  // namespace vision_deploy
