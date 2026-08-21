/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLO26_SEMANTIC_SEGMENTOR_H
#define YOLO26_SEMANTIC_SEGMENTOR_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class YOLO26SemanticSegmentor final : public vision_core::BaseModel {
public:
    YOLO26SemanticSegmentor(
        const std::string& model_path,
        int num_threads,
        int num_classes,
        bool lazy_load,
        std::string provider);

    void load_model() override;
    vision_core::InferResponse Run(
        const vision_core::InferRequest& request) override;
    std::vector<vision_core::InferIntent>
    supported_intents() const override;
    std::vector<vision_core::ModelCapability>
    get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node& config,
        bool lazy_load);

private:
    cv::Mat preprocess(const cv::Mat& bgr) const;
    std::vector<vision::Segmentation> segment(
        const vision_core::ImageInput& input);
    cv::Mat decode_label_map(
        const Ort::Value& output,
        const cv::Size& original_size) const;
    std::vector<vision::Segmentation> split_semantic_masks(
        const cv::Mat& label_map) const;

    int num_threads_;
    int num_classes_;
    std::string provider_;
    int input_height_ = 0;
    int input_width_ = 0;
};

}  // namespace vision_deploy

#endif  // YOLO26_SEMANTIC_SEGMENTOR_H
