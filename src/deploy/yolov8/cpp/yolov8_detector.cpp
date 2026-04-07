/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov8_detector.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> YOLOv8Detector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv8Detector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv8Detector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv8Detector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLOv8Detector::YOLOv8Detector(const std::string& model_path,
                                float conf_threshold,
                                float iou_threshold,
                                int num_threads,
                                bool lazy_load,
                                const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        num_classes_(0),
        num_threads_(num_threads),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void YOLOv8Detector::load_model() {
    init_session(num_threads_, provider_);
    if (output_num_ >= 2) {
        Ort::TypeInfo score_type_info = session_->GetOutputTypeInfo(1);
        auto score_tensor_info = score_type_info.GetTensorTypeAndShapeInfo();
        auto score_dims = score_tensor_info.GetShape();
        if (score_dims.size() >= 2 && score_dims[1] > 0) {
            num_classes_ = static_cast<int>(score_dims[1]);
        }
    }
    if (num_classes_ <= 0) {
        num_classes_ = 80;  // Default COCO classes
    }
    model_loaded_ = true;
}

cv::Mat YOLOv8Detector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);


    // Use common letterbox function (similar to Python implementation)
    cv::Mat padded = vision_common::letterbox(image,
                                            std::make_pair(inputHeight, inputWidth));


    return cv::dnn::blobFromImage(padded, 1.0/255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}



std::vector<vision_common::Result> YOLOv8Detector::detect(const cv::Mat& image) {
    ensure_model_loaded();
    reset_runtime_profile();
    const auto t0 = std::chrono::steady_clock::now();

    cv::Size orig_size = image.size();
    const auto t_pre0 = std::chrono::steady_clock::now();
    cv::Mat inputTensor = preprocess(image);
    const auto t_pre1 = std::chrono::steady_clock::now();
    set_runtime_preprocess_ms(std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());

    const auto t_infer0 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = run_session(inputTensor);
    const auto t_infer1 = std::chrono::steady_clock::now();
    set_runtime_model_infer_ms(std::chrono::duration<double, std::milli>(t_infer1 - t_infer0).count());

    const auto t_post0 = std::chrono::steady_clock::now();
    std::vector<vision_common::Result> results = postprocess(outputs, orig_size);
    const auto t_post1 = std::chrono::steady_clock::now();
    set_runtime_postprocess_ms(std::chrono::duration<double, std::milli>(t_post1 - t_post0).count());

    const auto t1 = std::chrono::steady_clock::now();
    set_runtime_total_ms(std::chrono::duration<double, std::milli>(t1 - t0).count());
    return results;
}

std::vector<vision_core::ModelCapability> YOLOv8Detector::get_capabilities() const {
    return {
        vision_core::ModelCapability::kImageInput,
        vision_core::ModelCapability::kDraw};
}

// DFL decode now uses common function

void YOLOv8Detector::get_dets(const cv::Size& orig_size, const float* boxes,
                                const float* scores, const float* score_sum,
                                const std::vector<int64_t>& dims,
                                int tensor_width, int tensor_height,
                                std::vector<vision_common::Result>& objects) {
    int grid_w = static_cast<int>(dims[2]);
    int grid_h = static_cast<int>(dims[3]);
    int anchors_per_branch = grid_w * grid_h;
    float scale_w = static_cast<float>(tensor_width) / static_cast<float>(grid_w);
    float scale_h = static_cast<float>(tensor_height) / static_cast<float>(grid_h);

    int orig_height = orig_size.height;
    int orig_width = orig_size.width;
    float scale2orign = std::min(
        static_cast<float>(tensor_height) / static_cast<float>(orig_width),
        static_cast<float>(tensor_width) / static_cast<float>(orig_height));
    int pad_h = static_cast<int>((tensor_width - orig_height * scale2orign) / 2);
    int pad_w = static_cast<int>((tensor_height - orig_width * scale2orign) / 2);

    for (int anchor_idx = 0; anchor_idx < anchors_per_branch; anchor_idx++) {
        if (score_sum[anchor_idx] < conf_threshold_) {
            continue;
        }

        // Get max score and class
        float max_score = -1.0f;
        int classId = -1;
        for (int class_idx = 0; class_idx < num_classes_; class_idx++) {
            size_t score_offset = class_idx * anchors_per_branch + anchor_idx;
            if ((scores[score_offset] > conf_threshold_) && (scores[score_offset] > max_score)) {
                max_score = scores[score_offset];
                classId = class_idx;
            }
        }

        if (classId >= 0) {
            // Use common dfl_decode function
            auto [x1, y1, x2, y2] = vision_common::dfl_decode(boxes, anchor_idx,
                anchors_per_branch, grid_h, scale_w, scale_h, scale2orign, pad_w, pad_h);
            vision_common::Result result;
            result.x1 = x1;
            result.y1 = y1;
            result.x2 = x2;
            result.y2 = y2;
            result.label = classId;
            result.score = max_score;
            result.keypoints.clear();  // Empty for detection
            objects.push_back(result);
        }
    }
}

std::vector<vision_common::Result> YOLOv8Detector::postprocess(
    std::vector<Ort::Value>& outputs,
    const cv::Size& orig_size) {

    std::vector<vision_common::Result> objects;

    // Process each output branch (3 branches with 3 outputs each: boxes, scores, score_sum)
    size_t output_num = outputs.size();
    int default_branch = 3;

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    for (int i = 0; i < static_cast<int>(output_num / default_branch); i++) {
        const float* boxes = outputs[i * default_branch].GetTensorMutableData<float>();
        const float* scores = outputs[i * default_branch + 1].GetTensorMutableData<float>();
        const float* score_sum = outputs[i * default_branch + 2].GetTensorMutableData<float>();

        std::vector<int64_t> dims = outputs[i * default_branch].GetTensorTypeAndShapeInfo().GetShape();

        // Note: swap width and height as in original code (inputHeight, inputWidth)
        get_dets(orig_size, boxes, scores, score_sum, dims, inputHeight, inputWidth, objects);
    }
    // Apply multi-class NMS using common function
    return vision_common::multi_class_nms(objects, iou_threshold_);
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv8Detector> registrar("YOLOv8Detector");

}  // namespace vision_deploy

