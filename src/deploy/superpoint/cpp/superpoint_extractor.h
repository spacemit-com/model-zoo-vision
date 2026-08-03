/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SUPERPOINT_EXTRACTOR_H
#define SUPERPOINT_EXTRACTOR_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

vision::LocalFeatures build_superpoint_features(
    const float* scores,
    const float* descriptor_map,
    int image_height,
    int image_width,
    int descriptor_channels,
    int descriptor_height,
    int descriptor_width,
    int num_keypoints,
    int nms_radius,
    int remove_borders,
    int original_width,
    int original_height,
    const std::string& feature_type);

class SuperPointExtractor final : public vision_core::BaseModel {
public:
    SuperPointExtractor(
        const std::string& model_path,
        int num_keypoints,
        int nms_radius,
        int remove_borders,
        std::string feature_type,
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
    vision::LocalFeatures extract(
        const vision_core::ImageInput& input);

    int num_keypoints_;
    int nms_radius_;
    int remove_borders_;
    std::string feature_type_;
    int num_threads_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // SUPERPOINT_EXTRACTOR_H
