/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolop.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

#include "operators/image_preprocess/cpu_image_preprocessor.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {
namespace {

constexpr int kInputHeight = 288;
constexpr int kInputWidth = 512;
constexpr int kLevels = 3;
constexpr int kAnchorsPerCell = 3;
constexpr int kAttributes = 6;
constexpr int kStrides[kLevels] = {8, 16, 32};
constexpr float kAnchors[kLevels][kAnchorsPerCell][2] = {
    {{3.0F, 9.0F}, {5.0F, 11.0F}, {4.0F, 20.0F}},
    {{7.0F, 18.0F}, {6.0F, 39.0F}, {12.0F, 31.0F}},
    {{19.0F, 50.0F}, {38.0F, 81.0F}, {68.0F, 157.0F}},
};
constexpr std::array<float, 3> kMean = {0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kStd = {0.229F, 0.224F, 0.225F};
constexpr const char* kExpectedOutputs[] = {
    "det_head_p3", "det_head_p4", "det_head_p5",
    "drive_area_seg", "lane_line_seg"};

double elapsed_ms(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

vision_operators::ImagePreprocessSpec make_preprocess_spec() {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = kInputWidth;
    spec.output_height = kInputHeight;
    spec.resize_mode = vision_operators::PreprocessResizeMode::kLetterbox;
    spec.output_rgb = false;
    spec.interpolation = vision_operators::PreprocessInterpolation::kArea;
    spec.padding = {114.0F, 114.0F, 114.0F};
    return spec;
}

vision_operators::CpuChannelTransform make_cpu_transform() {
    vision_operators::CpuChannelTransform transform;
    transform.input_divisor = {255.0F, 255.0F, 255.0F};
    transform.mean = kMean;
    transform.output_divisor = kStd;
    return transform;
}

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

float area(const vision::Detection& detection) {
    return std::max(0.0F, detection.bbox.x2 - detection.bbox.x1) *
        std::max(0.0F, detection.bbox.y2 - detection.bbox.y1);
}

std::vector<vision::Detection> decode_detections(
    std::vector<Ort::Value>& outputs,
    const YOLOP::Geometry& geometry,
    float conf_threshold,
    float iou_threshold,
    int max_det) {
    std::vector<vision::Detection> candidates;
    candidates.reserve(256);
    for (int level = 0; level < kLevels; ++level) {
        const auto info = outputs[static_cast<size_t>(level)]
            .GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> shape = info.GetShape();
        if (shape.size() != 4 || shape[0] != 1 ||
            shape[1] != kAnchorsPerCell * kAttributes ||
            shape[2] <= 0 || shape[3] <= 0) {
            throw std::runtime_error(
                "YOLOP: invalid detection head shape at level " +
                std::to_string(level));
        }
        const int height = static_cast<int>(shape[2]);
        const int width = static_cast<int>(shape[3]);
        const size_t plane = static_cast<size_t>(height) * width;
        const float* data = outputs[static_cast<size_t>(level)]
            .GetTensorData<float>();
        for (int anchor = 0; anchor < kAnchorsPerCell; ++anchor) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const size_t spatial = static_cast<size_t>(y) * width + x;
                    const auto value = [&](int attribute) {
                        const size_t channel =
                            static_cast<size_t>(anchor * kAttributes + attribute);
                        return sigmoid(data[channel * plane + spatial]);
                    };
                    const float objectness = value(4);
                    if (!(objectness > conf_threshold)) continue;
                    const float score = objectness * value(5);
                    if (!(score > conf_threshold)) continue;
                    const float center_x =
                        (value(0) * 2.0F - 0.5F + x) * kStrides[level];
                    const float center_y =
                        (value(1) * 2.0F - 0.5F + y) * kStrides[level];
                    const float scaled_w = value(2) * 2.0F;
                    const float scaled_h = value(3) * 2.0F;
                    const float box_w = scaled_w * scaled_w *
                        kAnchors[level][anchor][0];
                    const float box_h = scaled_h * scaled_h *
                        kAnchors[level][anchor][1];
                    vision::Detection detection;
                    detection.bbox = {
                        center_x - box_w * 0.5F,
                        center_y - box_h * 0.5F,
                        center_x + box_w * 0.5F,
                        center_y + box_h * 0.5F};
                    detection.score = score;
                    detection.label = 0;
                    candidates.push_back(std::move(detection));
                }
            }
        }
    }

    std::stable_sort(
        candidates.begin(), candidates.end(),
        [](const vision::Detection& left, const vision::Detection& right) {
            return left.score > right.score;
        });
    std::vector<vision::Detection> kept;
    kept.reserve(std::min(static_cast<size_t>(max_det), candidates.size()));
    for (const vision::Detection& candidate : candidates) {
        bool suppressed = false;
        for (const vision::Detection& selected : kept) {
            const float x1 = std::max(candidate.bbox.x1, selected.bbox.x1);
            const float y1 = std::max(candidate.bbox.y1, selected.bbox.y1);
            const float x2 = std::min(candidate.bbox.x2, selected.bbox.x2);
            const float y2 = std::min(candidate.bbox.y2, selected.bbox.y2);
            const float intersection =
                std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
            const float overlap = intersection /
                std::max(area(candidate) + area(selected) - intersection, 1.0e-12F);
            if (overlap > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) continue;
        kept.push_back(candidate);
        if (static_cast<int>(kept.size()) >= max_det) break;
    }

    const float max_x = static_cast<float>(geometry.original_width);
    const float max_y = static_cast<float>(geometry.original_height);
    for (vision::Detection& detection : kept) {
        detection.bbox.x1 = std::round(std::clamp(
            (detection.bbox.x1 - geometry.pad_w) / geometry.ratio,
            0.0F, max_x));
        detection.bbox.y1 = std::round(std::clamp(
            (detection.bbox.y1 - geometry.pad_h) / geometry.ratio,
            0.0F, max_y));
        detection.bbox.x2 = std::round(std::clamp(
            (detection.bbox.x2 - geometry.pad_w) / geometry.ratio,
            0.0F, max_x));
        detection.bbox.y2 = std::round(std::clamp(
            (detection.bbox.y2 - geometry.pad_h) / geometry.ratio,
            0.0F, max_y));
    }
    return kept;
}

cv::Mat decode_mask(
    Ort::Value& output,
    const YOLOP::Geometry& geometry) {
    const auto info = output.GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> shape = info.GetShape();
    if (shape.size() != 4 || shape[0] != 1 || shape[1] < 2 ||
        shape[2] != kInputHeight || shape[3] != kInputWidth) {
        throw std::runtime_error("YOLOP: invalid segmentation output shape");
    }
    const int channels = static_cast<int>(shape[1]);
    const size_t plane = static_cast<size_t>(kInputHeight) * kInputWidth;
    const float* data = output.GetTensorData<float>();
    const int top = static_cast<int>(std::lround(geometry.pad_h - 0.1F));
    const int left = static_cast<int>(std::lround(geometry.pad_w - 0.1F));
    const int height = std::min(geometry.resized_height, kInputHeight - top);
    const int width = std::min(geometry.resized_width, kInputWidth - left);
    cv::Mat content(height, width, CV_8UC1);
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            uint8_t* destination = content.ptr<uint8_t>(y);
            const size_t row = static_cast<size_t>(top + y) * kInputWidth + left;
            for (int x = 0; x < width; ++x) {
                int best = 0;
                float best_value = data[row + x];
                for (int channel = 1; channel < channels; ++channel) {
                    const float value = data[static_cast<size_t>(channel) * plane + row + x];
                    if (value > best_value) {
                        best_value = value;
                        best = channel;
                    }
                }
                destination[x] = best == 1 ? 255 : 0;
            }
        }
    });
    cv::Mat mask;
    cv::resize(
        content, mask,
        cv::Size(geometry.original_width, geometry.original_height),
        0.0, 0.0, cv::INTER_NEAREST);
    return mask;
}

vision::Segmentation mask_result(cv::Mat mask, int label) {
    vision::Segmentation result;
    result.bbox = {-1.0F, -1.0F, -1.0F, -1.0F};
    result.score = 1.0F;
    result.label = label;
    result.mask = std::make_shared<cv::Mat>(std::move(mask));
    return result;
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> YOLOP::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOP");
    }
    const YAML::Node params = config["default_params"];
    if (!params || !params.IsMap()) {
        throw std::runtime_error("default_params not found in config for YOLOP");
    }
    return std::make_unique<YOLOP>(
        model_path,
        vision_core::yaml_utils::getFloat(params, "conf_threshold", 0.25F),
        vision_core::yaml_utils::getFloat(params, "iou_threshold", 0.45F),
        vision_core::yaml_utils::getInt(params, "max_det", 300),
        vision_core::yaml_utils::getInt(params, "num_threads", 8),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

YOLOP::YOLOP(
    const std::string& model_path,
    float conf_threshold,
    float iou_threshold,
    int max_det,
    int num_threads,
    bool lazy_load,
    const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        max_det_(max_det),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) load_model();
}

void YOLOP::load_model() {
    if (model_loaded_) return;
    init_session(num_threads_, provider_);
    if (input_shape_.size() != 4 || input_shape_[0] != 1 ||
        input_shape_[1] != 3 || input_shape_[2] != kInputHeight ||
        input_shape_[3] != kInputWidth) {
        throw std::runtime_error("YOLOP expects input shape [1,3,288,512]");
    }
    if (output_num_ != 5) {
        throw std::runtime_error("YOLOP expects exactly 5 outputs");
    }
    for (size_t i = 0; i < output_num_; ++i) {
        if (output_names_[i] != kExpectedOutputs[i]) {
            throw std::runtime_error(
                "YOLOP output " + std::to_string(i) + " is '" +
                output_names_[i] + "', expected '" + kExpectedOutputs[i] + "'");
        }
    }
    model_loaded_ = true;
}

cv::Mat YOLOP::preprocess_cpu(
    const cv::Mat& image,
    Geometry* geometry) const {
    if (image.empty() || image.type() != CV_8UC3 || geometry == nullptr) {
        throw std::invalid_argument("YOLOP expects a non-empty BGR8 image");
    }
    geometry->original_width = image.cols;
    geometry->original_height = image.rows;
    geometry->ratio = std::min(
        static_cast<float>(kInputWidth) / image.cols,
        static_cast<float>(kInputHeight) / image.rows);
    geometry->resized_width = static_cast<int>(
        std::lround(image.cols * geometry->ratio));
    geometry->resized_height = static_cast<int>(
        std::lround(image.rows * geometry->ratio));
    geometry->pad_w = (kInputWidth - geometry->resized_width) * 0.5F;
    geometry->pad_h = (kInputHeight - geometry->resized_height) * 0.5F;
    return vision_operators::preprocess_bgr_to_nchw(
        image,
        make_preprocess_spec(),
        make_cpu_transform());
}

vision_core::InferResponse YOLOP::infer_input(
    const vision_core::ImageInput& input,
    float conf_threshold,
    float iou_threshold,
    int max_det) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto begin = std::chrono::steady_clock::now();
    Geometry geometry;
    const vision_operators::ImagePreprocessSpec spec =
        make_preprocess_spec();

    const auto preprocess_begin = std::chrono::steady_clock::now();
    auto prepared = prepare_image(
        input, spec,
        [this, &geometry](const cv::Mat& bgr) {
            return preprocess_cpu(bgr, &geometry);
        });
    const auto preprocess_end = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(elapsed_ms(preprocess_begin, preprocess_end));

    const auto infer_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(prepared.tensor());
    const auto infer_end = std::chrono::steady_clock::now();
    prepared.complete();
    set_runtime_model_infer_ms(elapsed_ms(infer_begin, infer_end));

    const auto postprocess_begin = std::chrono::steady_clock::now();
    std::vector<vision::Detection> detections = decode_detections(
        outputs, geometry, conf_threshold, iou_threshold, max_det);
    cv::Mat drivable = decode_mask(outputs[3], geometry);
    cv::Mat lane = decode_mask(outputs[4], geometry);
    vision_core::InferResponse response;
    response.results.reserve(detections.size() + 2);
    for (vision::Detection& detection : detections) {
        response.results.emplace_back(std::move(detection));
    }
    response.results.emplace_back(mask_result(std::move(drivable), 1));
    response.results.emplace_back(mask_result(std::move(lane), 2));
    const auto postprocess_end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(elapsed_ms(postprocess_begin, postprocess_end));
    set_runtime_total_ms(elapsed_ms(begin, postprocess_end));
    return response;
}

vision_core::InferResponse YOLOP::Run(
    const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kSegment) {
        return {{}, false, "YOLOP supports only kSegment"};
    }
    const auto* input = std::get_if<vision_core::ImageInput>(&request.input);
    if (input == nullptr) {
        return {{}, false, "YOLOP expects ImageInput"};
    }
    const float conf = request.params.conf_threshold > 0.0F
        ? request.params.conf_threshold : conf_threshold_;
    const float iou = request.params.iou_threshold > 0.0F
        ? request.params.iou_threshold : iou_threshold_;
    const int max_det = request.params.max_det > 0
        ? request.params.max_det : max_det_;
    return infer_input(*input, conf, iou, max_det);
}

std::vector<vision_core::InferIntent> YOLOP::supported_intents() const {
    return {vision_core::InferIntent::kSegment};
}

std::vector<vision_core::ModelCapability> YOLOP::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<YOLOP> registrar("YOLOP");

}  // namespace vision_deploy
