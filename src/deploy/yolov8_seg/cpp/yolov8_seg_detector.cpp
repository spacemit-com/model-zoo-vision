/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "yolov8_seg_detector.h"

#include <algorithm>
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

    cv::Size orig_size = image.size();

    // Preprocess
    cv::Mat inputTensor = preprocess(image);

    // Run inference using base class method
    std::vector<Ort::Value> outputs = run_session(inputTensor);

    // Postprocess
    std::vector<vision_common::Result> results = postprocess(outputs, orig_size);

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

        // Match results with original objects to get mask coefficients
        // (stored in embedding field temporarily)
        std::vector<std::vector<float>> mask_coeffs_list;
        for (const auto& res : results) {
            // Find matching object in original objects list
            bool found = false;
            for (const auto& obj : objects) {
                if (std::abs(obj.x1 - res.x1) < 1.0f &&
                    std::abs(obj.y1 - res.y1) < 1.0f &&
                    std::abs(obj.x2 - res.x2) < 1.0f &&
                    std::abs(obj.y2 - res.y2) < 1.0f &&
                    obj.label == res.label &&
                    std::abs(obj.score - res.score) < 0.01f) {
                    // Found match - use embedding field which contains seg_part coefficients
                    mask_coeffs_list.push_back(obj.embedding);
                    found = true;
                    break;
                }
            }
            if (!found) {
                // If no match found, create empty coefficients (shouldn't happen)
                mask_coeffs_list.push_back(std::vector<float>(proto_channels_, 0.0f));
            }
        }

        // Process masks
        std::vector<std::shared_ptr<cv::Mat>> masks = _process_masks(
            output_proto, proto_dims, mask_coeffs_list, results, orig_size);

        // Assign masks to results
        for (size_t i = 0; i < results.size() && i < masks.size(); i++) {
            results[i].mask = masks[i];
        }

        // Clear embedding field (it was only used to store seg_part temporarily)
        for (auto& res : results) {
            res.embedding.clear();
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

    // Squeeze batch dimension: [1, mask_dim, mask_h, mask_w] -> [mask_dim, mask_h, mask_w]
    int mask_dim = static_cast<int>(proto_dims[1]);
    int mask_h = static_cast<int>(proto_dims[2]);
    int mask_w = static_cast<int>(proto_dims[3]);

    // Flatten protos: [mask_dim, mask_h, mask_w] -> [mask_dim, mask_h * mask_w]
    // protos is stored as [1, mask_dim, mask_h, mask_w], so we skip the first dimension
    int proto_size = mask_h * mask_w;

    // Compute masks: mask_coeffs @ protos_flat -> [num_masks, mask_h * mask_w]
    int num_masks = static_cast<int>(mask_coeffs.size());
    std::vector<cv::Mat> masks(num_masks);

    for (int i = 0; i < num_masks; i++) {
        cv::Mat mask_flat(1, proto_size, CV_32F);
        float* mask_data = mask_flat.ptr<float>();

        // Matrix multiplication: mask_coeffs[i] @ protos_flat
        // protos layout: [1, mask_dim, mask_h, mask_w] -> [k * mask_h * mask_w + h * mask_w + w]
        for (int j = 0; j < proto_size; j++) {
            int h = j / mask_w;
            int w = j % mask_w;
            float sum = 0.0f;
            for (int k = 0; k < mask_dim; k++) {
                // protos[k * mask_h * mask_w + h * mask_w + w] = protos[k * proto_size + j]
                int proto_idx = k * mask_h * mask_w + h * mask_w + w;
                sum += mask_coeffs[i][k] * protos[proto_idx];
            }
            mask_data[j] = sum;
        }

        // Reshape to [mask_h, mask_w]
        masks[i] = mask_flat.reshape(0, mask_h);

        // Apply sigmoid using vision_common::sigmoid
        for (int y = 0; y < mask_h; y++) {
            for (int x = 0; x < mask_w; x++) {
                masks[i].at<float>(y, x) = vision_common::sigmoid(masks[i].at<float>(y, x));
            }
        }
    }

    // Calculate crop parameters (same as Python)
    int orig_h = orig_shape.height;
    int orig_w = orig_shape.width;
    float gain = std::min(static_cast<float>(mask_h) / orig_h, static_cast<float>(mask_w) / orig_w);
    float pad_w = mask_w - orig_w * gain;
    float pad_h = mask_h - orig_h * gain;
    pad_w /= 2.0f;
    pad_h /= 2.0f;

    int top = (pad_h > 0) ? static_cast<int>(std::round(pad_h - 0.1f)) : 0;
    int left = (pad_w > 0) ? static_cast<int>(std::round(pad_w - 0.1f)) : 0;
    int bottom = (pad_h > 0) ? (mask_h - static_cast<int>(std::round(pad_h + 0.1f))) : mask_h;
    int right = (pad_w > 0) ? (mask_w - static_cast<int>(std::round(pad_w + 0.1f))) : mask_w;

    top = std::max(0, std::min(top, mask_h - 1));
    left = std::max(0, std::min(left, mask_w - 1));
    bottom = std::max(top + 1, std::min(bottom, mask_h));
    right = std::max(left + 1, std::min(right, mask_w));

    // Crop and resize masks
    std::vector<std::shared_ptr<cv::Mat>> masks_scaled;
    for (int i = 0; i < num_masks; i++) {
        cv::Mat mask_cropped;
        if (top < bottom && left < right) {
            mask_cropped = masks[i](cv::Range(top, bottom), cv::Range(left, right));
        } else {
            mask_cropped = masks[i].clone();
        }

        // Resize to original image size
        cv::Mat mask_resized;
        cv::resize(mask_cropped, mask_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

        // Crop to bounding box
        const auto& result = results[i];
        int x1 = std::max(0, std::min(static_cast<int>(result.x1), orig_w - 1));
        int y1 = std::max(0, std::min(static_cast<int>(result.y1), orig_h - 1));
        int x2 = std::max(x1 + 1, std::min(static_cast<int>(result.x2), orig_w));
        int y2 = std::max(y1 + 1, std::min(static_cast<int>(result.y2), orig_h));

        cv::Mat mask_cropped_final = cv::Mat::zeros(orig_h, orig_w, CV_32F);
        mask_resized(cv::Range(y1, y2), cv::Range(x1, x2)).copyTo(
            mask_cropped_final(cv::Range(y1, y2), cv::Range(x1, x2)));

        // Threshold at 0.5 and convert to CV_8U (0 or 255)
        cv::Mat mask_binary;
        cv::threshold(mask_cropped_final, mask_binary, 0.5f, 255.0f, cv::THRESH_BINARY);

        // Convert to CV_8U type for OpenCV operations
        cv::Mat mask_uint8;
        mask_binary.convertTo(mask_uint8, CV_8U);

        masks_scaled.push_back(std::make_shared<cv::Mat>(mask_uint8));
    }

    return masks_scaled;
}

// Self-registration (runs at program startup)
static vision_core::ModelRegistrar<YOLOv8SegDetector> registrar("YOLOv8SegDetector");

}  // namespace vision_deploy

