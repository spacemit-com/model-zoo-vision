/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLO26_DEPTH_ESTIMATOR_H
#define YOLO26_DEPTH_ESTIMATOR_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class YOLO26DepthEstimator final : public vision_core::BaseModel {
public:
    YOLO26DepthEstimator(
        const std::string& model_path,
        int num_threads,
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
    vision::DepthMap estimate_depth(
        const vision_core::ImageInput& input);
    cv::Mat restore_depth(
        const Ort::Value& output,
        const cv::Size& original_size) const;

    int num_threads_;
    std::string provider_;
    int input_height_ = 0;
    int input_width_ = 0;
};

}  // namespace vision_deploy

#endif  // YOLO26_DEPTH_ESTIMATOR_H
