/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AV_TRACKER_H
#define AV_TRACKER_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class AVTracker final : public vision_core::BaseModel {
public:
    AVTracker(
        const std::string& model_path,
        int num_threads,
        bool lazy_load,
        std::string provider);

    void load_model() override;
    vision_core::InferResponse Run(
        const vision_core::InferRequest& request) override;
    std::vector<vision_core::InferIntent> supported_intents() const override;
    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node& config,
        bool lazy_load);

private:
    vision::Tracking initialize(
        const cv::Mat& image,
        const vision::BoundingBox& initial_box);
    vision::Tracking track(const cv::Mat& image);

    static constexpr int kTemplateSize = 128;
    static constexpr int kSearchSize = 256;
    static constexpr float kTemplateFactor = 2.0f;
    static constexpr float kSearchFactor = 4.0f;
    static constexpr float kMinimumBoxSize = 10.0f;

    int num_threads_;
    std::string provider_;
    bool initialized_ = false;
    cv::Rect2f state_box_;
    std::vector<float> template_tensor_;
};

}  // namespace vision_deploy

#endif  // AV_TRACKER_H
