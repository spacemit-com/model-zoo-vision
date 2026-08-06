/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MOBILESEG_H
#define MOBILESEG_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

cv::Size validate_mobileseg_image_input(
    const vision_core::ImageInput& input);

cv::Mat decode_mobileseg_label_map(
    const int32_t* labels,
    int model_height,
    int model_width,
    const cv::Size& original_size,
    int num_classes);

std::vector<vision::Segmentation>
split_mobileseg_semantic_masks(
    const cv::Mat& label_map,
    int num_classes);

class MobileSeg final : public vision_core::BaseModel {
public:
    MobileSeg(
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

    int num_threads_;
    int num_classes_;
    std::string provider_;
    int input_height_ = 0;
    int input_width_ = 0;
};

}  // namespace vision_deploy

#endif  // MOBILESEG_H
