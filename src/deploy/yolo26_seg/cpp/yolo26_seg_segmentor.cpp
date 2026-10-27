/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolo26_seg_segmentor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/cpp/image_processing.h"
#include "common/cpp/nms.h"
#include "operators/image_preprocess/image_preprocess_geometry.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {
namespace {

constexpr int kCocoClasses = 80;

cv::Size validate_image_input(const vision_core::ImageInput& input) {
    if (input.image.empty()) {
        throw std::invalid_argument("YOLO26-Seg input image is empty");
    }
    if (input.format == vision_core::ImagePixelFormat::kBgr8) {
        if (input.image.type() != CV_8UC3) {
            throw std::invalid_argument(
                "YOLO26-Seg BGR8 input must be CV_8UC3");
        }
        return input.image.size();
    }
    const int height = input.image.rows * 2 / 3;
    if (input.image.type() != CV_8UC1 ||
        input.image.rows % 3 != 0 ||
        (input.image.cols & 1) != 0 || (height & 1) != 0) {
        throw std::invalid_argument(
            "YOLO26-Seg NV12 input must be CV_8UC1 H*3/2 x W "
            "with even H and W");
    }
    return cv::Size(input.image.cols, height);
}

vision_operators::ImagePreprocessSpec make_preprocess_spec(
    int width, int height) {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = width;
    spec.output_height = height;
    spec.resize_mode = vision_operators::PreprocessResizeMode::kLetterbox;
    spec.output_rgb = true;
    spec.scale = {1.0F / 255.0F, 1.0F / 255.0F, 1.0F / 255.0F};
    spec.padding = {114.0F, 114.0F, 114.0F};
    return spec;
}

struct Candidate {
    vision_common::SegmentationResult result;
    const float* coefficients = nullptr;
};

}  // namespace

YOLO26SegSegmentor::YOLO26SegSegmentor(
    const std::string& model_path,
    float conf_threshold,
    float mask_threshold,
    int num_threads,
    bool lazy_load,
    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        mask_threshold_(mask_threshold),
        num_threads_(num_threads),
        provider_(provider) {
    if (!std::isfinite(conf_threshold_) ||
        conf_threshold_ < 0.0F || conf_threshold_ > 1.0F ||
        !std::isfinite(mask_threshold_) ||
        mask_threshold_ <= 0.0F || mask_threshold_ > 1.0F ||
        num_threads_ <= 0) {
        throw std::invalid_argument(
            "YOLO26-Seg conf must be in [0,1], mask in (0,1], and "
            "num_threads must be positive");
    }
    enable_accelerated_image_preprocess();
    if (!lazy_load) {
        load_model();
    }
}

std::unique_ptr<vision_core::BaseModel> YOLO26SegSegmentor::create(
    const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(
        config, {"conf_threshold", "mask_threshold"});
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("YOLO26-Seg model_path is missing");
    }
    const YAML::Node params = config["default_params"];
    return std::make_unique<YOLO26SegSegmentor>(
        model_path,
        vision_core::yaml_utils::getFloat(
            params, "conf_threshold", 0.25F),
        vision_core::yaml_utils::getFloat(
            params, "mask_threshold", 0.5F),
        vision_core::yaml_utils::getInt(params, "num_threads", 8),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

void YOLO26SegSegmentor::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_names_.size() != 1 || output_names_.size() != 2) {
        throw std::runtime_error(
            "YOLO26-Seg requires one input and two outputs");
    }
    const Ort::TypeInfo image_type = session_->GetInputTypeInfo(0);
    const Ort::TypeInfo detections_type = session_->GetOutputTypeInfo(0);
    const Ort::TypeInfo prototypes_type = session_->GetOutputTypeInfo(1);
    const auto image_info = image_type.GetTensorTypeAndShapeInfo();
    const auto detections_info =
        detections_type.GetTensorTypeAndShapeInfo();
    const auto prototypes_info =
        prototypes_type.GetTensorTypeAndShapeInfo();
    const auto detection_shape = detections_info.GetShape();
    const auto prototype_shape = prototypes_info.GetShape();
    if (image_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        input_shape_.size() != 4 || input_shape_[0] != 1 ||
        input_shape_[1] != 3 || input_shape_[2] <= 0 ||
        input_shape_[3] <= 0 ||
        input_shape_[2] > std::numeric_limits<int>::max() ||
        input_shape_[3] > std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            "YOLO26-Seg input must be float32 [1,3,H,W]");
    }
    if (detections_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        prototypes_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        detection_shape.size() != 3 || detection_shape[0] != 1 ||
        detection_shape[1] <= 0 ||
        prototype_shape.size() != 4 || prototype_shape[0] != 1 ||
        prototype_shape[1] <= 0 || prototype_shape[1] > 256 ||
        prototype_shape[2] <= 0 || prototype_shape[3] <= 0 ||
        prototype_shape[2] > std::numeric_limits<int>::max() ||
        prototype_shape[3] > std::numeric_limits<int>::max() ||
        prototype_shape[2] >
            std::numeric_limits<int>::max() / prototype_shape[3] ||
        detection_shape[2] != 6 + prototype_shape[1]) {
        throw std::runtime_error(
            "YOLO26-Seg outputs must be float32 [1,N,6+C] "
            "and [1,C,Hm,Wm]");
    }
    input_height_ = static_cast<int>(input_shape_[2]);
    input_width_ = static_cast<int>(input_shape_[3]);
    num_candidates_ = detection_shape[1];
    proto_channels_ = static_cast<int>(prototype_shape[1]);
    proto_height_ = static_cast<int>(prototype_shape[2]);
    proto_width_ = static_cast<int>(prototype_shape[3]);
    model_loaded_ = true;
}

cv::Mat YOLO26SegSegmentor::preprocess(const cv::Mat& bgr) const {
    return vision_common::letterbox_to_nchw_rgb_blob(
        bgr, {input_height_, input_width_});
}

vision_common::SegmentationResultList
YOLO26SegSegmentor::postprocess(
    const std::vector<Ort::Value>& outputs,
    const cv::Size& original_size,
    float conf_threshold,
    float mask_threshold,
    int max_det) const {
    if (outputs.size() != 2 || !outputs[0].IsTensor() ||
        !outputs[1].IsTensor()) {
        throw std::runtime_error(
            "YOLO26-Seg inference must return two tensors");
    }
    const auto detections_info =
        outputs[0].GetTensorTypeAndShapeInfo();
    const auto prototypes_info =
        outputs[1].GetTensorTypeAndShapeInfo();
    if (detections_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        prototypes_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        detections_info.GetShape() !=
            std::vector<int64_t>{1, num_candidates_,
                6 + proto_channels_} ||
        prototypes_info.GetShape() !=
            std::vector<int64_t>{1, proto_channels_,
                proto_height_, proto_width_}) {
        throw std::runtime_error(
            "YOLO26-Seg runtime output contract changed");
    }
    vision_common::mask_logit_threshold(mask_threshold);
    if (max_det == 0) {
        return {};
    }
    const auto geometry =
        vision_operators::make_image_preprocess_geometry(
            make_preprocess_spec(input_width_, input_height_),
            original_size.width, original_size.height);
    if (geometry.dst_width <= 0 || geometry.dst_height <= 0) {
        throw std::runtime_error("YOLO26-Seg letterbox geometry is empty");
    }
    const float scale_x =
        static_cast<float>(original_size.width) / geometry.dst_width;
    const float scale_y =
        static_cast<float>(original_size.height) / geometry.dst_height;
    const float* rows = outputs[0].GetTensorData<float>();
    const int features = 6 + proto_channels_;
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(num_candidates_));
    for (int64_t index = 0; index < num_candidates_; ++index) {
        const float* row = rows + index * features;
        if (!std::isfinite(row[4]) || row[4] < conf_threshold ||
            !std::isfinite(row[5]) ||
            !std::all_of(row, row + 4,
                [](float value) { return std::isfinite(value); })) {
            continue;
        }
        if (row[5] < 0.0F || row[5] >= kCocoClasses) {
            continue;
        }
        const int label = static_cast<int>(std::round(row[5]));
        if (label < 0 || label >= kCocoClasses ||
            std::fabs(row[5] - label) > 1.0e-3F ||
            !std::all_of(row + 6, row + features,
                [](float value) { return std::isfinite(value); })) {
            continue;
        }
        const float x1 = std::clamp(
            (row[0] - geometry.dst_x) * scale_x,
            0.0F, static_cast<float>(original_size.width));
        const float y1 = std::clamp(
            (row[1] - geometry.dst_y) * scale_y,
            0.0F, static_cast<float>(original_size.height));
        const float x2 = std::clamp(
            (row[2] - geometry.dst_x) * scale_x,
            0.0F, static_cast<float>(original_size.width));
        const float y2 = std::clamp(
            (row[3] - geometry.dst_y) * scale_y,
            0.0F, static_cast<float>(original_size.height));
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        Candidate candidate;
        candidate.result.bbox = {x1, y1, x2, y2};
        candidate.result.score = row[4];
        candidate.result.label = label;
        candidate.coefficients = row + 6;
        candidates.push_back(std::move(candidate));
    }
    vision_common::limit_by_score(
        candidates, max_det,
        [](const Candidate& candidate) {
            return candidate.result.score;
        });
    if (candidates.empty()) {
        return {};
    }

    cv::Mat coefficients(
        static_cast<int>(candidates.size()), proto_channels_, CV_32F);
    for (size_t index = 0; index < candidates.size(); ++index) {
        std::copy_n(
            candidates[index].coefficients, proto_channels_,
            coefficients.ptr<float>(static_cast<int>(index)));
    }
    const cv::Mat prototypes(
        proto_channels_, proto_height_ * proto_width_, CV_32F,
        const_cast<float*>(outputs[1].GetTensorData<float>()));
    cv::Mat logits;
    cv::gemm(coefficients, prototypes, 1.0, cv::Mat(), 0.0, logits);

    vision_common::SegmentationResultList results;
    results.reserve(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        cv::Mat mask_logits(
            proto_height_, proto_width_, CV_32F,
            logits.ptr<float>(static_cast<int>(index)));
        cv::Mat mask = vision_common::restore_instance_mask(
            mask_logits, candidates[index].result.bbox,
            original_size, mask_threshold);
        if (cv::countNonZero(mask) == 0) {
            continue;
        }
        candidates[index].result.mask =
            std::make_shared<cv::Mat>(std::move(mask));
        results.push_back(std::move(candidates[index].result));
    }
    return results;
}

vision_common::SegmentationResultList
YOLO26SegSegmentor::segment_input(
    const vision_core::ImageInput& input,
    float conf_threshold,
    float mask_threshold,
    int max_det) {
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

    const auto inference_start = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(prepared.tensor());
    const auto inference_end = std::chrono::steady_clock::now();
    prepared.complete();
    set_runtime_model_infer_ms(
        std::chrono::duration<double, std::milli>(
            inference_end - inference_start).count());

    const auto postprocess_start = std::chrono::steady_clock::now();
    auto results = postprocess(
        outputs, original_size, conf_threshold,
        mask_threshold, max_det);
    const auto postprocess_end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_start).count());
    set_runtime_total_ms(
        std::chrono::duration<double, std::milli>(
            postprocess_end - total_start).count());
    return results;
}

vision_core::InferResponse YOLO26SegSegmentor::RunImpl(
    const vision_core::InferRequest& request) {
    const auto& image_input =
        std::get<vision_core::ImageInput>(request.input);
    const float conf_threshold =
        vision_core::resolve_probability_threshold(
            request.params.conf_threshold,
            conf_threshold_, "conf_threshold");
    const float mask_threshold =
        vision_core::resolve_probability_threshold(
            request.params.mask_threshold,
            mask_threshold_, "mask_threshold");
    // The one-to-one head is NMS-free; the public IoU parameter is unused.
    vision_core::resolve_probability_threshold(
        request.params.iou_threshold, 0.5F, "iou_threshold");
    auto results = segment_input(
        image_input, conf_threshold,
        mask_threshold, request.params.max_det);
    return vision_core::make_infer_response(std::move(results));
}

std::vector<vision_core::InferIntent>
YOLO26SegSegmentor::supported_intents() const {
    return {vision_core::InferIntent::kSegment};
}

std::vector<vision_core::ModelCapability>
YOLO26SegSegmentor::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<YOLO26SegSegmentor> registrar(
    "YOLO26SegSegmentor");

}  // namespace vision_deploy
