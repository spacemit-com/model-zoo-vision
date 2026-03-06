/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RESNET_CLASSIFIER_H
#define RESNET_CLASSIFIER_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief ResNet Image Classifier
 * 
 * Classifies images using ResNet model.
 */
class ResNetClassifier : public vision_core::BaseModel, public vision_core::IClassificationModel {
public:
    ResNetClassifier(const std::string& model_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~ResNetClassifier() = default;

    /**
     * @brief Load ResNet ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for ResNet inference
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    std::vector<vision_common::Result> classify(const cv::Mat& image) override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    std::vector<vision_common::Result> postprocess(std::vector<Ort::Value>& outputs);

private:
    int num_threads_;
    cv::Size resize_size_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // RESNET_CLASSIFIER_H
