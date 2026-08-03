/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LIGHTGLUE_MATCHER_H
#define LIGHTGLUE_MATCHER_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

std::string validate_lightglue_features(
    const vision::LocalFeatures& features,
    const std::string& expected_feature_type,
    int expected_keypoints,
    int expected_descriptor_dim);

std::vector<vision::FeatureMatch> filter_lightglue_matches(
    const float* log_scores,
    int keypoint_count,
    const vision::LocalFeatures& query,
    const vision::LocalFeatures& train,
    float filter_threshold);

class LightGlueMatcher final : public vision_core::BaseModel {
public:
    LightGlueMatcher(
        const std::string& model_path,
        std::string feature_type,
        int num_keypoints,
        int descriptor_dim,
        float filter_threshold,
        int num_threads,
        bool lazy_load,
        std::string provider);

    void load_model() override;
    vision_core::InferResponse Run(
        const vision_core::InferRequest& request) override;
    std::vector<vision_core::InferIntent> supported_intents() const override;

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node& config,
        bool lazy_load);

private:
    std::vector<vision::FeatureMatch> match(
        const vision_core::LocalFeaturePairInput& input);

    std::string feature_type_;
    int num_keypoints_;
    int descriptor_dim_;
    float filter_threshold_;
    int num_threads_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // LIGHTGLUE_MATCHER_H
