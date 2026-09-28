/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RTMPOSE_ESTIMATOR_H
#define RTMPOSE_ESTIMATOR_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

// Top-down COCO 17-keypoint model. Accepts a single-person BGR image of any
// size and returns keypoints in that image's coordinates.
class RTMPoseEstimator : public vision_core::BaseModel {
public:
    RTMPoseEstimator(const std::string& model_path, int num_threads, const std::string& provider,
        float confidence_threshold, bool lazy_load);

    void load_model() override;
    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;
    std::vector<vision_core::InferIntent> supported_intents() const override;
    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

private:
    int num_threads_;
    std::string provider_;
    float confidence_threshold_;
    size_t simcc_x_index_ = 0;
    size_t simcc_y_index_ = 1;
};

}  // namespace vision_deploy

#endif  // RTMPOSE_ESTIMATOR_H
