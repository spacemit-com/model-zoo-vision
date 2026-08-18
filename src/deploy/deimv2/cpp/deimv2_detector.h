/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DEIMV2_DETECTOR_H
#define DEIMV2_DETECTOR_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class DEIMv2Detector : public vision_core::BaseModel,
                        public vision_core::IDetectionModel {
public:
    DEIMv2Detector(
        const std::string& model_path,
        float conf_threshold = 0.4F,
        int num_threads = 8,
        bool normalize = false,
        bool lazy_load = false,
        const std::string& provider = "SpaceMITExecutionProvider");

    ~DEIMv2Detector() override = default;

    void load_model() override;

    vision_common::DetectionResultList detect(
        const cv::Mat& image,
        float conf_threshold = -1.0F,
        float iou_threshold = -1.0F) override;

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
    struct LetterboxGeometry {
        float ratio = 1.0F;
        int left = 0;
        int top = 0;
    };

    cv::Mat preprocess(const cv::Mat& image) const;
    std::vector<Ort::Value> run_session_two_inputs(
        const cv::Mat& image_tensor);
    vision_common::DetectionResultList detect_input(
        const vision_core::ImageInput& input,
        float conf_threshold);
    vision_common::DetectionResultList postprocess(
        std::vector<Ort::Value>& outputs,
        const cv::Size& original_size,
        const LetterboxGeometry& geometry,
        float conf_threshold) const;
    LetterboxGeometry calculate_geometry(
        const cv::Size& original_size) const;

    float conf_threshold_;
    int num_threads_;
    bool normalize_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // DEIMV2_DETECTOR_H
