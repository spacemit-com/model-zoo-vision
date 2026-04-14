/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov8_seg_detector.h"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "vision_model_config.h"
#include "vision_model_factory.h"

namespace vision_deploy {

std::unique_ptr<vision_core::BaseModel> YOLOv8SegDetector::create(const YAML::Node& config, bool lazy_load) {
    std::string model_path = vision_core::yaml_utils::getString(config, "model_path");
    if (model_path.empty()) {
        throw std::runtime_error("model_path not found in config for YOLOv8SegDetector");
    }

    YAML::Node default_params = config["default_params"];
    if (!default_params) {
        throw std::runtime_error("default_params not found in config for YOLOv8SegDetector");
    }

    float conf_threshold = vision_core::yaml_utils::getFloat(default_params, "conf_threshold", 0.25f);
    float iou_threshold = vision_core::yaml_utils::getFloat(default_params, "iou_threshold", 0.45f);
    int num_threads = vision_core::yaml_utils::getInt(default_params, "num_threads", 4);
    std::string provider = vision_core::yaml_utils::getProvider(config);

    return std::make_unique<YOLOv8SegDetector>(
        model_path, conf_threshold, iou_threshold, num_threads, lazy_load, provider);
}

YOLOv8SegDetector::YOLOv8SegDetector(const std::string& model_path,
                                        float conf_threshold,
                                        float iou_threshold,
                                        int num_threads,
                                        bool lazy_load,
                                        const std::string& provider)
    : BaseModel(model_path, lazy_load),
        conf_threshold_(conf_threshold),
        iou_threshold_(iou_threshold),
        num_threads_(num_threads),
        num_classes_(0),
        proto_channels_(0),
        provider_(provider) {
    if (!lazy_load) {
        load_model();
    }
}

void YOLOv8SegDetector::load_model() {
    init_session(num_threads_, provider_);

    // Extract num_classes and proto_channels from output shapes
    // Output structure: [box0, score0, sum0, box1, score1, sum1, box2, score2, sum2, seg0, seg1, seg2, proto]
    // score output shape: [1, num_classes, h, w]
    if (output_num_ >= 2) {
        Ort::TypeInfo score_type_info = session_->GetOutputTypeInfo(1);
        auto score_tensor_info = score_type_info.GetTensorTypeAndShapeInfo();
        auto score_dims = score_tensor_info.GetShape();
        if (score_dims.size() >= 2 && score_dims[1] > 0) {
            num_classes_ = static_cast<int>(score_dims[1]);
        }
    }

    // proto output shape: [1, proto_channels, mask_h, mask_w]
    if (output_num_ >= 13) {
        Ort::TypeInfo proto_type_info = session_->GetOutputTypeInfo(12);
        auto proto_tensor_info = proto_type_info.GetTensorTypeAndShapeInfo();
        auto proto_dims = proto_tensor_info.GetShape();
        if (proto_dims.size() >= 2 && proto_dims[1] > 0) {
            proto_channels_ = static_cast<int>(proto_dims[1]);
        }
    }

    // Default values if not found
    if (num_classes_ <= 0) {
        num_classes_ = 80;  // Default COCO classes
    }
    if (proto_channels_ <= 0) {
        proto_channels_ = 32;  // Default proto channels
    }

    model_loaded_ = true;
}

cv::Mat YOLOv8SegDetector::preprocess(const cv::Mat& image) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    ensure_model_loaded();

    int inputWidth = static_cast<int>(input_shape_[3]);  // width
    int inputHeight = static_cast<int>(input_shape_[2]);  // height

    // Use letterbox preprocessing (same as Python)
    cv::Mat padded = vision_common::letterbox(image,
                                            std::make_pair(inputHeight, inputWidth));

    return cv::dnn::blobFromImage(padded, 1.0/255.0,
        cv::Size(inputWidth, inputHeight),
        cv::Scalar(0, 0, 0), true, false, CV_32F);
}

std::vector<vision_common::Result> YOLOv8SegDetector::segment(const cv::Mat& image) {
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

std::vector<vision_core::ModelCapability> YOLOv8SegDetector::get_capabilities() const {
    return {
        vision_core::ModelCapability::kImageInput,
        vision_core::ModelCapability::kDraw};
}

std::vector<vision_common::Result> YOLOv8SegDetector::postprocess(std::vector<Ort::Value>& outputs,
                                                                const cv::Size& orig_size) {
    std::vector<vision_common::Result> objects;

    int inputWidth = static_cast<int>(input_shape_[3]);
    int inputHeight = static_cast<int>(input_shape_[2]);

    // Process 3 branches (8x, 16x, 32x downsampling)
    for (int i = 0; i < 3; i++) {
        const float* boxes = outputs[i * 3].GetTensorMutableData<float>();
        const float* scores = outputs[i * 3 + 1].GetTensorMutableData<float>();
        const float* score_sum = outputs[i * 3 + 2].GetTensorMutableData<float>();
        const float* seg_part = outputs[9 + i].GetTensorMutableData<float>();
        std::vector<int64_t> dims = outputs[i * 3].GetTensorTypeAndShapeInfo().GetShape();
        Get_Dets(orig_size, boxes, scores, score_sum, seg_part, dims, inputHeight, inputWidth,
                num_classes_, objects);
    }

    // Apply multi-class NMS
    std::vector<vision_common::Result> results = vision_common::multi_class_nms(objects, iou_threshold_);

    // Process masks if we have results and proto output
    if (!results.empty() && outputs.size() >= 13) {
        const float* output_proto = outputs[12].GetTensorMutableData<float>();
        std::vector<int64_t> proto_dims = outputs[12].GetTensorTypeAndShapeInfo().GetShape();

        // Extract mask coefficients directly from results (preserved through NMS via embedding field)
        std::vector<std::vector<float>> mask_coeffs_list;
        mask_coeffs_list.reserve(results.size());
        for (const auto& res : results) {
            mask_coeffs_list.push_back(res.embedding);
        }

        // Process masks
        std::vector<std::shared_ptr<cv::Mat>> masks = _process_masks(
            output_proto, proto_dims, mask_coeffs_list, results, orig_size);

        // Assign masks and clear temporary embedding field
        for (size_t i = 0; i < results.size(); i++) {
            if (i < masks.size()) {
                results[i].mask = masks[i];
            }
            results[i].embedding.clear();
        }
    }

    return results;
}

void YOLOv8SegDetector::Get_Dets(const cv::Size& orig_size, const float* boxes,
                                    const float* scores, const float* score_sum,
                                    const float* seg_part, std::vector<int64_t> dims,
                                    int tensor_width, int tensor_height, int num_classes,
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
        for (int class_idx = 0; class_idx < num_classes; class_idx++) {
            size_t score_offset = class_idx * anchors_per_branch + anchor_idx;
            if (scores[score_offset] > conf_threshold_ && scores[score_offset] > max_score) {
                max_score = scores[score_offset];
                classId = class_idx;
            }
        }

        if (classId >= 0) {  // detect object
            auto [x1, y1, x2, y2] = vision_common::dfl_decode(boxes, anchor_idx, anchors_per_branch,
                grid_w, scale_w, scale_h, scale2orign, pad_w, pad_h);
            vision_common::Result result;

            // Store seg_part coefficients in embedding field temporarily (32 coefficients)
            result.embedding.clear();
            result.embedding.reserve(proto_channels_);
            for (int i = 0; i < proto_channels_; i++) {
                result.embedding.push_back(seg_part[i * anchors_per_branch + anchor_idx]);
            }

            result.x1 = x1;
            result.y1 = y1;
            result.x2 = x2;
            result.y2 = y2;
            result.label = classId;
            result.score = max_score;
            objects.push_back(result);
        }
    }
}

std::vector<std::shared_ptr<cv::Mat>> YOLOv8SegDetector::_process_masks(
    const float* protos,
    const std::vector<int64_t>& proto_dims,
    const std::vector<std::vector<float>>& mask_coeffs,
    const std::vector<vision_common::Result>& results,
    const cv::Size& orig_shape) {
    if (results.empty() || mask_coeffs.empty()) {
        return std::vector<std::shared_ptr<cv::Mat>>();
    }

    int mask_dim = static_cast<int>(proto_dims[1]);
    int mask_h = static_cast<int>(proto_dims[2]);
    int mask_w = static_cast<int>(proto_dims[3]);
    int proto_size = mask_h * mask_w;
    int num_masks = static_cast<int>(mask_coeffs.size());

    // Manual matmul: [num_masks, mask_dim] x [mask_dim, proto_size] = [num_masks, proto_size]
    // Outer loop over k (channels) for cache-friendly sequential reads of proto rows.
    // ~7x faster than Eigen GEMM on RISC-V which lacks SIMD optimizations.
    std::vector<float> raw_masks(static_cast<size_t>(num_masks) * proto_size, 0.0f);
    for (int k = 0; k < mask_dim; ++k) {
        const float* proto_row = protos + k * proto_size;
        for (int i = 0; i < num_masks; ++i) {
            float c = mask_coeffs[i][k];
            float* out = raw_masks.data() + i * proto_size;
            for (int j = 0; j < proto_size; ++j) {
                out[j] += c * proto_row[j];
            }
        }
    }
    // Skip sigmoid: sigmoid(x) > 0.5 <==> x > 0, threshold on raw logits

    // Calculate crop parameters for letterbox padding removal
    int orig_h = orig_shape.height;
    int orig_w = orig_shape.width;
    float gain = std::min(static_cast<float>(mask_h) / orig_h, static_cast<float>(mask_w) / orig_w);
    float pad_w = (mask_w - orig_w * gain) / 2.0f;
    float pad_h = (mask_h - orig_h * gain) / 2.0f;

    int top = (pad_h > 0) ? static_cast<int>(std::round(pad_h - 0.1f)) : 0;
    int left = (pad_w > 0) ? static_cast<int>(std::round(pad_w - 0.1f)) : 0;
    int bottom = (pad_h > 0) ? (mask_h - static_cast<int>(std::round(pad_h + 0.1f))) : mask_h;
    int right = (pad_w > 0) ? (mask_w - static_cast<int>(std::round(pad_w + 0.1f))) : mask_w;

    top = std::max(0, std::min(top, mask_h - 1));
    left = std::max(0, std::min(left, mask_w - 1));
    bottom = std::max(top + 1, std::min(bottom, mask_h));
    right = std::max(left + 1, std::min(right, mask_w));

    const int crop_h = bottom - top;
    const int crop_w = right - left;
    const float scale_x = static_cast<float>(crop_w) / static_cast<float>(orig_w);
    const float scale_y = static_cast<float>(crop_h) / static_cast<float>(orig_h);

    std::vector<std::shared_ptr<cv::Mat>> masks_scaled;
    masks_scaled.reserve(num_masks);

    for (int i = 0; i < num_masks; ++i) {
        // Each mask is contiguous in [i * proto_size, (i+1) * proto_size).
        cv::Mat mask_full(mask_h, mask_w, CV_32F, raw_masks.data() + static_cast<size_t>(i) * proto_size);

        // Crop letterbox padding
        cv::Mat mask_cropped = mask_full(cv::Range(top, bottom), cv::Range(left, right));

        // Map bbox from original image coords to small-mask coords
        const auto& result = results[i];
        int bx1 = std::max(0, static_cast<int>(std::floor(result.x1 * scale_x)));
        int by1 = std::max(0, static_cast<int>(std::floor(result.y1 * scale_y)));
        int bx2 = std::min(crop_w, static_cast<int>(std::ceil(result.x2 * scale_x)));
        int by2 = std::min(crop_h, static_cast<int>(std::ceil(result.y2 * scale_y)));

        if (bx2 <= bx1 || by2 <= by1) {
            masks_scaled.push_back(std::make_shared<cv::Mat>(
                cv::Mat::zeros(orig_h, orig_w, CV_8U)));
            continue;
        }

        // Crop bbox region from small mask, resize only that region
        cv::Mat small_roi = mask_cropped(cv::Range(by1, by2), cv::Range(bx1, bx2));

        int ox1 = std::max(0, std::min(static_cast<int>(result.x1), orig_w - 1));
        int oy1 = std::max(0, std::min(static_cast<int>(result.y1), orig_h - 1));
        int ox2 = std::max(ox1 + 1, std::min(static_cast<int>(result.x2), orig_w));
        int oy2 = std::max(oy1 + 1, std::min(static_cast<int>(result.y2), orig_h));

        cv::Mat roi_resized;
        cv::resize(small_roi, roi_resized,
                    cv::Size(ox2 - ox1, oy2 - oy1), 0, 0, cv::INTER_LINEAR);

        // Threshold on raw logits: x > 0 <==> sigmoid(x) > 0.5
        cv::Mat roi_binary;
        cv::threshold(roi_resized, roi_binary, 0.0f, 255.0f, cv::THRESH_BINARY);
        cv::Mat roi_uint8;
        roi_binary.convertTo(roi_uint8, CV_8U);

        // Paste into full-size output mask
        cv::Mat mask_out = cv::Mat::zeros(orig_h, orig_w, CV_8U);
        roi_uint8.copyTo(mask_out(cv::Range(oy1, oy2), cv::Range(ox1, ox2)));

        masks_scaled.push_back(std::make_shared<cv::Mat>(mask_out));
    }

    return masks_scaled;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv8SegDetector> registrar("YOLOv8SegDetector");

}  // namespace vision_deploy

