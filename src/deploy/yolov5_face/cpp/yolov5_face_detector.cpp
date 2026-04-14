/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov5_face_detector.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> YOLOv5FaceDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv5FaceDetector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv5FaceDetector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv5FaceDetector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLOv5FaceDetector::YOLOv5FaceDetector(const std::string& model_path,
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
    if (!lazy_load) {
        load_model();
    }
}

void YOLOv5FaceDetector::load_model() {
    init_session(num_threads_, provider_);
    model_loaded_ = true;
}

cv::Mat YOLOv5FaceDetector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);  // width
    int inputHeight = static_cast<int>(input_shape_[2]);  // height

    // Use common letterbox function (already converts BGR to RGB)
    cv::Mat padded = vision_common::letterbox(image,
                                            std::make_pair(inputHeight, inputWidth));
    return cv::dnn::blobFromImage(padded, 1.0/255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}

std::vector<vision_common::Result> YOLOv5FaceDetector::detect(const cv::Mat& image) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    cv::Size orig_size = image.size();

    // Preprocess
    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat inputTensor = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    // Run inference using base class method
    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(inputTensor);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    // Postprocess
    const auto t_post0 = std::chrono::steady_clock::now();
    std::vector<vision_common::Result> results = postprocess(outputs, orig_size);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());

    return results;
}

std::vector<vision_core::ModelCapability> YOLOv5FaceDetector::get_capabilities() const {
    return {
        vision_core::ModelCapability::kImageInput,
        vision_core::ModelCapability::kDraw};
}

std::vector<vision_common::Result> YOLOv5FaceDetector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size) {

    // Initialize anchors (same as Python code: reshape(3, 1, 3, 1, 1, 2))
    // Reshape to (3, 1, 3, 1, 1, 2) means: 3 scales, each with 3 anchors, each anchor has (w, h)
    // Simplified to (3, 3, 2) for easier access: [scale][anchor][w_or_h]
    const float init_anchors[3][3][2] = {
        {{4.0f, 5.0f}, {8.0f, 10.0f}, {13.0f, 16.0f}},
        {{23.0f, 29.0f}, {43.0f, 55.0f}, {73.0f, 105.0f}},
        {{146.0f, 217.0f}, {231.0f, 300.0f}, {335.0f, 433.0f}}
    };

    const int strides[3] = {8, 16, 32};

    // Optimized: Pre-compute logit threshold to avoid unnecessary sigmoid calculations
    // Reverse sigmoid: raw_conf_threshold = -log(1.0/conf_threshold - 1)
    // If raw value < threshold, sigmoid(raw) < conf_threshold, so we can skip
    float raw_conf_threshold = -std::log(1.0f / conf_threshold_ - 1.0f);

    // Optimized: Pre-allocate memory and use fixed-size struct instead of vector
    // Estimate total predictions: 3 scales * channels * ny * nx
    // Reserve space to avoid reallocations
    std::vector<Prediction> pred_list;
    size_t estimated_size = 0;

    // First pass: calculate total size for pre-allocation
    for (size_t idx = 0; idx < outputs.size() && idx < 3; ++idx) {
        auto tensor_info = outputs[idx].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();
        int channels = static_cast<int>(dims[1]);  // 3 anchors
        int ny = static_cast<int>(dims[2]);
        int nx = static_cast<int>(dims[3]);
        estimated_size += channels * ny * nx;
    }
    pred_list.reserve(estimated_size);

    // Process each output (3 scales)
    for (size_t idx = 0; idx < outputs.size() && idx < 3; ++idx) {
        const float* pred_data = outputs[idx].GetTensorMutableData<float>();
        auto tensor_info = outputs[idx].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensor_info.GetShape();

        int bs = static_cast<int>(dims[0]);
        int channels = static_cast<int>(dims[1]);  // 3 anchors
        int ny = static_cast<int>(dims[2]);
        int nx = static_cast<int>(dims[3]);
        int features = static_cast<int>(dims[4]);  // 16 features

        // Create grid
        std::vector<float> grid = make_grid(nx, ny);

        // Process predictions for batch 0
        for (int anchor = 0; anchor < channels; ++anchor) {
            for (int h = 0; h < ny; ++h) {
                for (int w = 0; w < nx; ++w) {
                    size_t base_idx = anchor * ny * nx * features +
                                        h * nx * features + w * features;

                    // Optimized: Early exit if obj_conf (index 4) is below threshold
                    // Compare raw logit value first to avoid expensive sigmoid calculation
                    float raw_obj_conf = pred_data[base_idx + 4];
                    if (raw_obj_conf < raw_conf_threshold) {
                        continue;  // Skip this prediction, no need to compute sigmoid
                    }

                    // Only compute sigmoid for predictions that pass the threshold check
                    // Same as Python: y_tmp = sigmoid(np.concatenate([pred[..., :5], pred[..., 15:]], axis=-1))
                    float x_tmp = vision_common::sigmoid(pred_data[base_idx]);
                    float y_tmp = vision_common::sigmoid(pred_data[base_idx + 1]);
                    float w_tmp = vision_common::sigmoid(pred_data[base_idx + 2]);
                    float h_tmp = vision_common::sigmoid(pred_data[base_idx + 3]);
                    float obj_conf = vision_common::sigmoid(raw_obj_conf);
                    float cls_conf = vision_common::sigmoid(pred_data[base_idx + 15]);  // Last feature

                    // Decode coordinates: (y_tmp * 2. - 0.5 + grid) * strides[idx]
                    // Python: y[..., :2] = (y_tmp[..., :2] * 2. - 0.5 + grid) * strides[idx]
                    // grid has shape (1, 1, ny, nx, 2), accessed as grid[h][w] = [x, y]
                    int grid_idx = (h * nx + w) * 2;
                    float x = (x_tmp * 2.0f - 0.5f + grid[grid_idx]) * strides[idx];
                    float y = (y_tmp * 2.0f - 0.5f + grid[grid_idx + 1]) * strides[idx];

                    // Decode width/height: (y_tmp[..., 2:4] * 2) ** 2 * init_anchors[idx]
                    // Python: y[..., 2:4] = (y_tmp[..., 2:4] * 2) ** 2 * init_anchors[idx]
                    // init_anchors[idx] has shape (1, 3, 1, 1, 2), simplified to (3, 2) for easier access
                    float w_box = std::pow(w_tmp * 2.0f, 2) * init_anchors[idx][anchor][0];
                    float h_box = std::pow(h_tmp * 2.0f, 2) * init_anchors[idx][anchor][1];

                    // Store: [x, y, w, h, obj_conf, cls_conf] in xywh format (same as Python)
                    // Python postprocess returns (bs, -1, 6) with [x, y, w, h, obj_conf, cls_conf]
                    // Conversion to xyxy happens in NMS
                    // Optimized: Use fixed-size struct instead of vector to avoid heap allocation
                    pred_list.emplace_back(x, y, w_box, h_box, obj_conf, cls_conf);
                }
            }
        }
    }

    // Apply NMS
    std::vector<vision_common::Result> results = non_max_suppression_face(pred_list);

    // Scale coordinates back to original image size
    // Python: scale_coords(self.input_shape, pred[0][i][:4], img_src.shape)
    // self.input_shape is shape[2:4] = [height, width] from Python
    // img_src.shape is (height, width, channels)
    // In C++: input_shape_ is [batch, channels, height, width] from ONNX
    // Python uses shape[2:4] which is indices 2 and 3 (height, width)
    // Same as yolov8_detector.cpp: input_shape_[2] = height, input_shape_[3] = width
    int inputHeight = static_cast<int>(input_shape_[2]);  // height (index 2)
    int inputWidth = static_cast<int>(input_shape_[3]);   // width (index 3)
    cv::Size input_shape(inputWidth, inputHeight);  // cv::Size is (width, height)

        for (auto& result : results) {
        float coords[4] = {result.x1, result.y1, result.x2, result.y2};
        vision_common::scale_coords(input_shape, coords, orig_size);
        result.x1 = coords[0];
        result.y1 = coords[1];
        result.x2 = coords[2];
        result.y2 = coords[3];
    }

    return results;
}

std::vector<vision_common::Result> YOLOv5FaceDetector::non_max_suppression_face(
    const std::vector<Prediction>& predictions) {

    if (predictions.empty()) {
        return std::vector<vision_common::Result>();
    }

    // First filter: obj_conf > conf_threshold (same as Python: xc = prediction[..., 4] > conf_thres)
    // Optimized: Pre-allocate filtered_preds to avoid reallocations
    std::vector<Prediction> filtered_preds;
    filtered_preds.reserve(predictions.size());
    for (const auto& pred : predictions) {
        if (pred.obj_conf > conf_threshold_) {  // obj_conf
            filtered_preds.push_back(pred);
        }
    }

    if (filtered_preds.empty()) {
        return std::vector<vision_common::Result>();
    }

    // Compute final confidence: conf = obj_conf * cls_conf (same as Python: x[:, 5:] *= x[:, 4:5])
    // Convert xywh to xyxy (same as Python: box = xywh2xyxy(x[:, :4]))
    // Optimized: Pre-allocate candidates to avoid reallocations
    std::vector<vision_common::Result> candidates;
    candidates.reserve(filtered_preds.size());
    for (const auto& pred : filtered_preds) {
        float final_conf = pred.obj_conf * pred.cls_conf;  // Final confidence

        // Second filter: final_conf > conf_threshold (same as Python: [conf.reshape(-1) > conf_thres])
        if (final_conf > conf_threshold_) {
            // Convert xywh to xyxy (same as Python: box = xywh2xyxy(x[:, :4]))
            float xywh[4] = {pred.x, pred.y, pred.w, pred.h};
            float xyxy[4];
            vision_common::xywh2xyxy(xywh, xyxy);

            vision_common::Result result;
            result.x1 = xyxy[0];
            result.y1 = xyxy[1];
            result.x2 = xyxy[2];
            result.y2 = xyxy[3];
            result.score = final_conf;  // Use final confidence
            result.label = -1;  // Face class (no label display)
            candidates.push_back(result);
        }
    }

    // Apply NMS using common function
    if (candidates.empty()) {
        return std::vector<vision_common::Result>();
    }

    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    boxes.reserve(candidates.size());
    scores.reserve(candidates.size());

    for (const auto& cand : candidates) {
        boxes.push_back(cv::Rect2f(cand.x1, cand.y1, cand.x2 - cand.x1, cand.y2 - cand.y1));
        scores.push_back(cand.score);
    }

    std::vector<int> keep_indices = vision_common::nms(boxes, scores, iou_threshold_);

    std::vector<vision_common::Result> final_results;
    final_results.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        final_results.push_back(candidates[idx]);
    }

    return final_results;
}

std::vector<float> YOLOv5FaceDetector::make_grid(int nx, int ny) {
    // Flat array: [w0, h0, w1, h1, ...] — 2 floats per grid cell
    std::vector<float> grid(static_cast<size_t>(ny * nx * 2));
    for (int h = 0; h < ny; ++h) {
        for (int w = 0; w < nx; ++w) {
            size_t idx = static_cast<size_t>((h * nx + w) * 2);
            grid[idx] = static_cast<float>(w);
            grid[idx + 1] = static_cast<float>(h);
        }
    }
    return grid;
}


// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv5FaceDetector> registrar("YOLOv5FaceDetector");

}  // namespace vision_deploy

