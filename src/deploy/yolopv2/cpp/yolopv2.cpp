/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolopv2.h"

#include <algorithm>
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
constexpr int kCanonicalHeight = 720;
constexpr int kCanonicalWidth = 1280;
constexpr int kLevels = 3;
constexpr int kAnchorsPerCell = 3;
constexpr int kAttributes = 85;
// The released YOLOPv2 checkpoint keeps the 80-class YOLOR head layout, but
// its single traffic-object class is stored in raw class slot 3.  Do not
// expose that implementation detail as a public class id (or interpret it as
// COCO class 3); the model contract is the same single "vehicle" class used
// by YOLOP.
constexpr int kVehicleOutputClass = 3;
constexpr int kVehicleLabel = 0;
constexpr int kDrivableAreaLabel = 1;
constexpr int kLaneLineLabel = 2;
constexpr int kStrides[kLevels] = {8, 16, 32};
constexpr float kAnchors[kLevels][kAnchorsPerCell][2] = {
    {{12.0F, 16.0F}, {19.0F, 36.0F}, {40.0F, 28.0F}},
    {{36.0F, 75.0F}, {76.0F, 55.0F}, {72.0F, 146.0F}},
    {{142.0F, 110.0F}, {192.0F, 243.0F}, {459.0F, 401.0F}},
};
constexpr const char* kExpectedOutputs[] = {
    "det_head_p3", "det_head_p4", "det_head_p5", "seg", "ll"};

double elapsed_ms(
    const std::chrono::steady_clock::time_point& begin,
    const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

float detection_area(const vision::Detection& detection) {
    return std::max(0.0F, detection.bbox.x2 - detection.bbox.x1) *
        std::max(0.0F, detection.bbox.y2 - detection.bbox.y1);
}

std::vector<vision::Detection> decode_detections(
    std::vector<Ort::Value>& outputs,
    const cv::Size& original_size,
    float conf_threshold,
    float iou_threshold,
    int max_det) {
    std::vector<vision::Detection> candidates;
    candidates.reserve(512);
    for (int level = 0; level < kLevels; ++level) {
        const auto info = outputs[static_cast<size_t>(level)]
            .GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> shape = info.GetShape();
        if (shape.size() != 4 || shape[0] != 1 ||
            shape[1] != kAnchorsPerCell * kAttributes ||
            shape[2] <= 0 || shape[3] <= 0) {
            throw std::runtime_error(
                "YOLOPv2: invalid detection head shape at level " +
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
                    const float best_score =
                        objectness * value(5 + kVehicleOutputClass);
                    if (!(best_score > conf_threshold)) continue;
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
                    detection.score = best_score;
                    detection.label = kVehicleLabel;
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
            if (candidate.label != selected.label) continue;
            const float x1 = std::max(candidate.bbox.x1, selected.bbox.x1);
            const float y1 = std::max(candidate.bbox.y1, selected.bbox.y1);
            const float x2 = std::min(candidate.bbox.x2, selected.bbox.x2);
            const float y2 = std::min(candidate.bbox.y2, selected.bbox.y2);
            const float intersection =
                std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
            const float overlap = intersection /
                std::max(
                    detection_area(candidate) + detection_area(selected) -
                        intersection,
                    1.0e-12F);
            if (overlap > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (suppressed) continue;
        kept.push_back(candidate);
        if (static_cast<int>(kept.size()) >= max_det) break;
    }

    const float scale_x =
        static_cast<float>(original_size.width) / kInputWidth;
    const float scale_y =
        static_cast<float>(original_size.height) / kInputHeight;
    const float max_x = static_cast<float>(original_size.width);
    const float max_y = static_cast<float>(original_size.height);
    for (vision::Detection& detection : kept) {
        detection.bbox.x1 = std::rint(std::clamp(
            detection.bbox.x1 * scale_x, 0.0F, max_x));
        detection.bbox.y1 = std::rint(std::clamp(
            detection.bbox.y1 * scale_y, 0.0F, max_y));
        detection.bbox.x2 = std::rint(std::clamp(
            detection.bbox.x2 * scale_x, 0.0F, max_x));
        detection.bbox.y2 = std::rint(std::clamp(
            detection.bbox.y2 * scale_y, 0.0F, max_y));
    }
    return kept;
}

std::pair<cv::Mat, cv::Mat> decode_masks(
    Ort::Value& drivable_output,
    Ort::Value& lane_output,
    const cv::Size& original_size) {
    const std::vector<int64_t> drivable_shape =
        drivable_output.GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> lane_shape =
        lane_output.GetTensorTypeAndShapeInfo().GetShape();
    if (drivable_shape != std::vector<int64_t>({1, 2, kInputHeight, kInputWidth}) ||
        lane_shape != std::vector<int64_t>({1, 1, kInputHeight, kInputWidth})) {
        throw std::runtime_error("YOLOPv2: invalid segmentation output shapes");
    }
    const size_t plane = static_cast<size_t>(kInputHeight) * kInputWidth;
    const float* drivable = drivable_output.GetTensorData<float>();
    const float* lane = lane_output.GetTensorData<float>();
    cv::Mat drivable_0(kInputHeight, kInputWidth, CV_32F,
        const_cast<float*>(drivable));
    cv::Mat drivable_1(kInputHeight, kInputWidth, CV_32F,
        const_cast<float*>(drivable + plane));
    cv::Mat lane_net(kInputHeight, kInputWidth, CV_32F,
        const_cast<float*>(lane));
    const cv::Size canonical_size(kCanonicalWidth, kCanonicalHeight);
    cv::Mat drivable_0_up;
    cv::Mat drivable_1_up;
    cv::Mat lane_up;
    cv::resize(drivable_0, drivable_0_up, canonical_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(drivable_1, drivable_1_up, canonical_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(lane_net, lane_up, canonical_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat drivable_canonical(canonical_size, CV_8UC1);
    cv::Mat lane_canonical(canonical_size, CV_8UC1);
    cv::parallel_for_(cv::Range(0, kCanonicalHeight), [&](const cv::Range& rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            const float* da0 = drivable_0_up.ptr<float>(y);
            const float* da1 = drivable_1_up.ptr<float>(y);
            const float* ll = lane_up.ptr<float>(y);
            uint8_t* da_mask = drivable_canonical.ptr<uint8_t>(y);
            uint8_t* ll_mask = lane_canonical.ptr<uint8_t>(y);
            for (int x = 0; x < kCanonicalWidth; ++x) {
                da_mask[x] = da1[x] > da0[x] ? 255 : 0;
                ll_mask[x] = std::rint(ll[x]) == 1.0F ? 255 : 0;
            }
        }
    });
    if (original_size == canonical_size) {
        return {
            std::move(drivable_canonical),
            std::move(lane_canonical)};
    }
    cv::Mat drivable_mask;
    cv::Mat lane_mask;
    cv::resize(
        drivable_canonical, drivable_mask, original_size,
        0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(
        lane_canonical, lane_mask, original_size,
        0.0, 0.0, cv::INTER_NEAREST);
    return {std::move(drivable_mask), std::move(lane_mask)};
}

vision::Segmentation mask_result(cv::Mat mask, int label) {
    vision::Segmentation result;
    result.bbox = {-1.0F, -1.0F, -1.0F, -1.0F};
    result.score = 1.0F;
    result.label = label;
    result.mask = std::make_shared<cv::Mat>(std::move(mask));
    return result;
}

vision_operators::ImagePreprocessSpec make_preprocess_spec() {
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = kInputWidth;
    spec.output_height = kInputHeight;
    spec.resize_mode = vision_operators::PreprocessResizeMode::kStretch;
    spec.output_rgb = true;
    spec.scale = {1.0F / 255.0F, 1.0F / 255.0F, 1.0F / 255.0F};
    spec.interpolation = vision_operators::PreprocessInterpolation::kBilinear;
    return spec;
}

}  // namespace

std::unique_ptr<vision_core::BaseModel> YOLOPv2::create(
    const YAML::Node& config,
    bool lazy_load) {
    const std::string model_path =
        vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOPv2");
    }
    const YAML::Node params = config["default_params"];
    if (!params || !params.IsMap()) {
        throw std::runtime_error("default_params not found in config for YOLOPv2");
    }
    return std::make_unique<YOLOPv2>(
        model_path,
        vision_core::yaml_utils::getFloat(params, "conf_threshold", 0.3F),
        vision_core::yaml_utils::getFloat(params, "iou_threshold", 0.45F),
        vision_core::yaml_utils::getInt(params, "max_det", 300),
        vision_core::yaml_utils::getInt(params, "num_threads", 8),
        lazy_load,
        vision_core::yaml_utils::getProvider(config));
}

YOLOPv2::YOLOPv2(
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

void YOLOPv2::load_model() {
    if (model_loaded_) return;
    init_session(num_threads_, provider_);
    if (input_shape_.size() != 4 || input_shape_[0] != 1 ||
        input_shape_[1] != 3 || input_shape_[2] != kInputHeight ||
        input_shape_[3] != kInputWidth) {
        throw std::runtime_error("YOLOPv2 expects input shape [1,3,288,512]");
    }
    if (output_num_ != 5) {
        throw std::runtime_error(
            "YOLOPv2 expects 5 outputs; the 8-output anchor-grid model is unsupported");
    }
    for (size_t i = 0; i < output_num_; ++i) {
        if (output_names_[i] != kExpectedOutputs[i]) {
            throw std::runtime_error(
                "YOLOPv2 output " + std::to_string(i) + " is '" +
                output_names_[i] + "', expected '" + kExpectedOutputs[i] + "'");
        }
    }
    model_loaded_ = true;
}

vision_core::InferResponse YOLOPv2::infer_input(
    const vision_core::ImageInput& input,
    float conf_threshold,
    float iou_threshold,
    int max_det) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto begin = std::chrono::steady_clock::now();
    const cv::Size original_size(
        input.image.cols,
        input.format == vision_core::ImagePixelFormat::kNv12
            ? input.image.rows * 2 / 3 : input.image.rows);
    const vision_operators::ImagePreprocessSpec spec = make_preprocess_spec();

    const auto preprocess_begin = std::chrono::steady_clock::now();
    auto prepared = prepare_image(
        input, spec,
        [&spec](const cv::Mat& bgr) {
            // Preserve the reference pipeline's two bilinear sampling stages.
            // Collapsing 1280x720 -> 512x288 into one direct resize changes
            // quantized detections and segmentation boundaries materially.
            thread_local cv::Mat canonical;
            const cv::Mat* source = &bgr;
            if (bgr.cols != kCanonicalWidth ||
                bgr.rows != kCanonicalHeight) {
                cv::resize(
                    bgr, canonical,
                    cv::Size(kCanonicalWidth, kCanonicalHeight),
                    0.0, 0.0, cv::INTER_LINEAR);
                source = &canonical;
            }
            return vision_operators::preprocess_bgr_to_nchw(*source, spec);
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
        outputs, original_size, conf_threshold, iou_threshold, max_det);
    auto masks = decode_masks(outputs[3], outputs[4], original_size);
    vision_core::InferResponse response;
    response.results.reserve(detections.size() + 2);
    for (vision::Detection& detection : detections) {
        response.results.emplace_back(std::move(detection));
    }
    response.results.emplace_back(
        mask_result(std::move(masks.first), kDrivableAreaLabel));
    response.results.emplace_back(
        mask_result(std::move(masks.second), kLaneLineLabel));
    const auto postprocess_end = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(elapsed_ms(postprocess_begin, postprocess_end));
    set_runtime_total_ms(elapsed_ms(begin, postprocess_end));
    return response;
}

vision_core::InferResponse YOLOPv2::Run(
    const vision_core::InferRequest& request) {
    if (request.intent != vision_core::InferIntent::kSegment) {
        return {{}, false, "YOLOPv2 supports only kSegment"};
    }
    const auto* input = std::get_if<vision_core::ImageInput>(&request.input);
    if (input == nullptr) {
        return {{}, false, "YOLOPv2 expects ImageInput"};
    }
    const float conf = request.params.conf_threshold > 0.0F
        ? request.params.conf_threshold : conf_threshold_;
    const float iou = request.params.iou_threshold > 0.0F
        ? request.params.iou_threshold : iou_threshold_;
    const int max_det = request.params.max_det > 0
        ? request.params.max_det : max_det_;
    return infer_input(*input, conf, iou, max_det);
}

std::vector<vision_core::InferIntent> YOLOPv2::supported_intents() const {
    return {vision_core::InferIntent::kSegment};
}

std::vector<vision_core::ModelCapability> YOLOPv2::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

static vision_core::ModelRegistrar<YOLOPv2> registrar("YOLOPv2");

}  // namespace vision_deploy
