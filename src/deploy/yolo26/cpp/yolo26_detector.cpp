/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolo26_detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/dnn.hpp>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {

}  // namespace

std::unique_ptr<vision_core::BaseModel> YOLO26Detector::create(const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(config, {"conf_threshold", "iou_threshold"});
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLO26Detector");
    }
    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLO26Detector");
    }
    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);
    return std::make_unique<YOLO26Detector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLO26Detector::YOLO26Detector(
    const std::string& model_path,
    float conf_threshold,
    float iou_threshold,
    int num_threads,
    bool lazy_load,
    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        num_threads_(num_threads),
        provider_(provider) {
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

void YOLO26Detector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_shape_.size() != 4 || input_shape_[0] == 0 || input_shape_[0] > 1 ||
        input_shape_[1] != 3 || input_shape_[2] <= 0 || input_shape_[3] <= 0) {
        throw std::runtime_error("YOLO26Detector requires single-image NCHW RGB input with fixed positive height and width");
    }
    model_loaded_ = true;
}

cv::Mat YOLO26Detector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();
    int input_width = static_cast<int>(input_shape_[3]);
    int input_height = static_cast<int>(input_shape_[2]);
    return vision_common::letterbox_to_nchw_rgb_blob(
        image,
        std::make_pair(input_height, input_width));
}

vision_common::DetectionResultList YOLO26Detector::detect_input(
    const vision_core::ImageInput& input,
    float conf_threshold,
    float iou_threshold, int max_det) {
    const float use_conf = vision_core::resolve_probability_threshold(
        conf_threshold, conf_threshold_, "conf_threshold");
    const float use_iou = vision_core::resolve_probability_threshold(
        iou_threshold, iou_threshold_, "iou_threshold");
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    const cv::Size orig_size(
        input.image.cols,
        input.format == vision_core::ImagePixelFormat::kNv12
            ? input.image.rows * 2 / 3
            : input.image.rows);
    const auto t_pre0 = std::chrono::steady_clock::now();
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = static_cast<int>(input_shape_[3]);
    spec.output_height = static_cast<int>(input_shape_[2]);
    spec.resize_mode =
        vision_operators::PreprocessResizeMode::kLetterbox;
    spec.output_rgb = true;
    spec.scale = {
        1.0F / 255.0F,
        1.0F / 255.0F,
        1.0F / 255.0F};
    spec.padding = {114.0F, 114.0F, 114.0F};
    auto prepared = prepare_image(
        input, spec,
        [this](const cv::Mat& bgr) {
            return preprocess(bgr);
        });
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs =
        run_session(prepared.tensor());
    const auto t_infer1 = std::chrono::steady_clock::now();
    prepared.complete();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    const auto t_post0 = std::chrono::steady_clock::now();
    vision_common::DetectionResultList results = postprocess(outputs, orig_size, use_conf, use_iou);
    vision_common::filter_and_limit_detections(results, max_det);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

vision_core::InferResponse YOLO26Detector::RunImpl(const vision_core::InferRequest& request) {
    const auto& image_input = std::get<vision_core::ImageInput>(request.input);
    vision_common::DetectionResultList detections =
        detect_input(
            image_input,
            request.params.conf_threshold,
            request.params.iou_threshold, request.params.max_det);
    return vision_core::make_infer_response(std::move(detections));
}

std::vector<vision_core::InferIntent> YOLO26Detector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

std::vector<vision_core::ModelCapability> YOLO26Detector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::DetectionResultList YOLO26Detector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold) {
    (void)iou_threshold;
    if (outputs.empty()) {
        return {};
    }
    if (outputs.size() != 1) {
        throw std::runtime_error("YOLO26 expects single output tensor");
    }

    if (!outputs[0] || !outputs[0].IsTensor() ||
        outputs[0].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("YOLO26 output must be float32");
    }

    const int input_height = static_cast<int>(input_shape_[2]);
    const int input_width = static_cast<int>(input_shape_[3]);
    Ort::Value& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = info.GetShape();
    if (dims.size() < 2) {
        throw std::runtime_error("Unexpected YOLO26 output shape (dims < 2)");
    }

    int64_t num_anchors = 0;
    int64_t features = 0;
    if (dims.size() == 3) {
        const int64_t dim1 = dims[1];
        const int64_t dim2 = dims[2];
        if (dims[0] != 1) {
            throw std::runtime_error("YOLO26 only supports batch size 1");
        }
        // only support [1, N, 6]
        if (dim2 != 6) {
            throw std::runtime_error("YOLO26 only supports [N,6] e2e output");
        }
        num_anchors = dim1;
        features = dim2;
    } else if (dims.size() == 2) {
        // support [N, 6]
        if (dims[1] != 6) {
            throw std::runtime_error("YOLO26 only supports [N,6] e2e output");
        }
        num_anchors = dims[0];
        features = dims[1];
    } else {
        throw std::runtime_error("Unexpected YOLO26 output rank");
    }

    const float* data = out.GetTensorData<float>();
    if (features != 6) {
        throw std::runtime_error("YOLO26 only supports [N,6] e2e output");
    }

    const float gain = std::min(
        static_cast<float>(input_height) / static_cast<float>(orig_size.height),
        static_cast<float>(input_width) / static_cast<float>(orig_size.width));
    const float pad_w = (static_cast<float>(input_width) - orig_size.width * gain) / 2.0f;
    const float pad_h = (static_cast<float>(input_height) - orig_size.height * gain) / 2.0f;
    const float max_x = static_cast<float>(std::max(orig_size.width - 1, 1));
    const float max_y = static_cast<float>(std::max(orig_size.height - 1, 1));

    vision_common::DetectionResultList results;
    results.reserve(static_cast<size_t>(num_anchors));
    for (int64_t i = 0; i < num_anchors; ++i) {
        const float* row = data + i * features;
        const float score = row[4];
        if (!std::isfinite(score) || score < conf_threshold ||
            !std::all_of(row, row + 4, [](float value) { return std::isfinite(value); }) ||
            !std::isfinite(row[5])) {
            continue;
        }

        const double label = std::round(static_cast<double>(row[5]));
        if (label < 0.0 || label > std::numeric_limits<int>::max()) {
            continue;
        }
        const int cls = static_cast<int>(label);

        float x1 = (row[0] - pad_w) / gain;
        float y1 = (row[1] - pad_h) / gain;
        float x2 = (row[2] - pad_w) / gain;
        float y2 = (row[3] - pad_h) / gain;
        x1 = std::clamp(x1, 0.0f, max_x);
        y1 = std::clamp(y1, 0.0f, max_y);
        x2 = std::clamp(x2, 0.0f, max_x);
        y2 = std::clamp(y2, 0.0f, max_y);
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        vision_common::DetectionResult r;
        r.bbox = vision_common::BoundingBox{x1, y1, x2, y2};
        r.score = score;
        r.label = cls;
        results.push_back(r);
    }
    return results;
}

static vision_core::ModelRegistrar<YOLO26Detector> registrar("YOLO26Detector");

}  // namespace vision_deploy
