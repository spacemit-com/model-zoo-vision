/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov5_detector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

namespace {
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
}  // namespace

std::unique_ptr<vision_core::BaseModel> YOLOv5Detector::create(const YAML::Node& config, bool lazy_load) {
    vision_core::yaml_utils::validateModelParams(config, {
        "conf_threshold", "iou_threshold", "has_objectness"});
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv5Detector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv5Detector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    auto model = std::make_unique<YOLOv5Detector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
    model->has_objectness_ = vision_core::yaml_utils::getBool(default_params, "has_objectness", true);
    return model;
}

YOLOv5Detector::YOLOv5Detector(const std::string& model_path,
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

void YOLOv5Detector::load_model() {
    if (model_loaded_) {
        return;
    }
    init_session(num_threads_, provider_);
    if (input_shape_.size() != 4 || input_shape_[0] == 0 || input_shape_[0] > 1 ||
        input_shape_[1] != 3 || input_shape_[2] <= 0 || input_shape_[3] <= 0) {
        throw std::runtime_error("YOLOv5Detector requires single-image NCHW RGB input with fixed positive height and width");
    }
    model_loaded_ = true;
}

cv::Mat YOLOv5Detector::preprocess(const cv::Mat& image) {
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

vision_common::DetectionResultList YOLOv5Detector::detect_input(
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


std::vector<vision_core::InferIntent> YOLOv5Detector::supported_intents() const {
    return {vision_core::InferIntent::kDetect};
}

vision_core::InferResponse YOLOv5Detector::RunImpl(const vision_core::InferRequest& request) {
    const auto& image_input = std::get<vision_core::ImageInput>(request.input);

    vision_common::DetectionResultList task_results =
        detect_input(
            image_input,
            request.params.conf_threshold,
            request.params.iou_threshold, request.params.max_det);
    return vision_core::make_infer_response(std::move(task_results));
}

std::vector<vision_core::ModelCapability> YOLOv5Detector::get_capabilities() const {
    return {vision_core::ModelCapability::kDraw};
}

vision_common::DetectionResultList YOLOv5Detector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size,
    float conf_threshold,
    float iou_threshold) {
    if (outputs.empty()) {
        return {};
    }

    if (outputs.size() != 1 || !outputs[0] || !outputs[0].IsTensor() ||
        outputs[0].GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("YOLOv5 expects one float32 output tensor");
    }

    const int input_height = static_cast<int>(input_shape_[2]);
    const int input_width = static_cast<int>(input_shape_[3]);

    Ort::Value& out0 = outputs[0];
    auto info = out0.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> dims = info.GetShape();
    if (dims.size() < 2) {
        throw std::runtime_error("Unexpected YOLOv5 output shape (dims < 2)");
    }

    int64_t num_boxes = 0;
    int64_t features = 0;
    bool layout_cn = false;
    if (dims.size() == 3) {
        if (dims[0] != 1) {
            throw std::runtime_error("YOLOv5 only supports batch size 1");
        }
        const int64_t dim1 = dims[1];
        const int64_t dim2 = dims[2];
        if ((dim1 <= 256 && dim2 > 256) || dim1 < dim2) {
            features = dim1;
            num_boxes = dim2;
            layout_cn = true;
        } else {
            num_boxes = dim1;
            features = dim2;
            layout_cn = false;
        }
    } else if (dims.size() == 2) {
        num_boxes = dims[0];
        features = dims[1];
    } else {
        throw std::runtime_error("YOLOv5 only supports rank-two or rank-three decoded outputs");
    }

    // Standard YOLOv5 exports use xywh + objectness + class probabilities,
    // including custom models with fewer than the 80 COCO classes.
    const bool has_objectness = has_objectness_.value_or(features >= 85);
    const int64_t cls_start = has_objectness ? 5 : 4;
    if (features <= cls_start) {
        throw std::runtime_error("YOLOv5 output must contain box, score and class channels");
    }
    const int64_t num_classes = features - cls_start;

    const float* data = out0.GetTensorData<float>();

    auto get_val = [&](int64_t box_idx, int64_t ch_idx) -> float {
        if (layout_cn) {
            return data[ch_idx * num_boxes + box_idx];
        }
        return data[box_idx * features + ch_idx];
    };

    bool need_sigmoid = false;
    bool need_scale_xywh = false;
    const int64_t probe_n = std::min<int64_t>(num_boxes, 200);
    float min_conf = 1e9f;
    float max_conf = -1e9f;
    float max_xywh = 0.0f;
    for (int64_t i = 0; i < probe_n; ++i) {
        for (int k = 0; k < 4; ++k) {
            max_xywh = std::max(max_xywh, std::fabs(get_val(i, k)));
        }
        if (has_objectness) {
            const float obj_raw = get_val(i, 4);
            if (std::isfinite(obj_raw)) {
                min_conf = std::min(min_conf, obj_raw);
                max_conf = std::max(max_conf, obj_raw);
            }
        }
        if (num_classes > 0) {
            const float cls_raw = get_val(i, cls_start);
            if (std::isfinite(cls_raw)) {
                min_conf = std::min(min_conf, cls_raw);
                max_conf = std::max(max_conf, cls_raw);
            }
        }
    }
    if (max_conf > 1.5f || min_conf < -0.1f) {
        need_sigmoid = true;
    }
    if (max_xywh <= 1.5f) {
        need_scale_xywh = true;
    }

    std::vector<vision_common::DetectionResult> candidates;
    candidates.reserve(static_cast<size_t>(num_boxes));

    for (int64_t i = 0; i < num_boxes; ++i) {
        float obj = 1.0f;
        if (has_objectness) {
            obj = get_val(i, 4);
            if (!std::isfinite(obj)) continue;
            if (need_sigmoid) {
                obj = sigmoid(obj);
            }
            if (obj <= conf_threshold) {
                continue;
            }
        }

        float best_raw = -std::numeric_limits<float>::infinity();
        int best_cls = -1;
        for (int64_t c = 0; c < num_classes; ++c) {
            const float raw = get_val(i, cls_start + c);
            if (std::isfinite(raw) && raw > best_raw) {
                best_raw = raw;
                best_cls = static_cast<int>(c);
            }
        }

        const float best_score = need_sigmoid
            ? sigmoid(best_raw) : best_raw;
        const float best_conf = has_objectness
            ? best_score * obj : best_score;

        if (best_cls < 0 ||
            !(best_conf > 0.0f) ||
            !(best_conf > conf_threshold)) {
            continue;
        }

        float xywh[4] = {get_val(i, 0), get_val(i, 1), get_val(i, 2), get_val(i, 3)};
        if (need_scale_xywh) {
            xywh[0] *= static_cast<float>(input_width);
            xywh[1] *= static_cast<float>(input_height);
            xywh[2] *= static_cast<float>(input_width);
            xywh[3] *= static_cast<float>(input_height);
        }

        float xyxy[4];
        vision_common::xywh2xyxy(xywh, xyxy);

        vision_common::DetectionResult r;
        r.bbox = vision_common::BoundingBox{xyxy[0], xyxy[1], xyxy[2], xyxy[3]};
        r.score = best_conf;
        r.label = best_cls;
        candidates.push_back(r);
    }

    if (candidates.empty()) {
        return {};
    }

    auto results = vision_common::multi_class_nms(candidates, iou_threshold);
    // Preserve the existing global score order after class-wise suppression.
    std::stable_sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.score > b.score; });

    const cv::Size input_shape(input_width, input_height);
    for (auto& r : results) {
        float coords[4] = {r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2};
        vision_common::scale_coords(input_shape, coords, orig_size);
        r.bbox.x1 = coords[0];
        r.bbox.y1 = coords[1];
        r.bbox.x2 = coords[2];
        r.bbox.y2 = coords[3];
    }

    return results;
}

static vision_core::ModelRegistrar<YOLOv5Detector> registrar("YOLOv5Detector");

}  // namespace vision_deploy
