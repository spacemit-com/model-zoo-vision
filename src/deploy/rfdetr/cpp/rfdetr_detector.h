/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RFDETR_DETECTOR_H
#define RFDETR_DETECTOR_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class RFDETRDetector final : public vision_core::BaseModel {
public:
    RFDETRDetector(
        const std::string& model_path,
        float conf_threshold = 0.3F,
        int max_det = 300,
        int num_threads = 8,
        bool lazy_load = false,
        const std::string& provider = "SpaceMITExecutionProvider");

    ~RFDETRDetector() override = default;

    void load_model() override;

    vision_core::InferResponse Run(
        const vision_core::InferRequest& request) override;
    std::vector<vision_core::InferIntent> supported_intents()
        const override;
    std::vector<vision_core::ModelCapability> get_capabilities()
        const override;

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node& config,
        bool lazy_load);

private:
    cv::Mat preprocess(const cv::Mat& bgr) const;
    vision_common::DetectionResultList infer_input(
        const vision_core::ImageInput& input,
        float conf_threshold,
        int max_det);
    vision_common::DetectionResultList postprocess(
        const std::vector<Ort::Value>& outputs,
        const cv::Size& original_size,
        float conf_threshold,
        int max_det) const;

    float conf_threshold_;
    int max_det_;
    int num_threads_;
    int input_width_ = 0;
    int input_height_ = 0;
    int64_t num_queries_ = 0;
    int64_t num_classes_ = 0;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // RFDETR_DETECTOR_H
