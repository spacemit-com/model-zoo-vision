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

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief ResNet Image Classifier
 * 
 * Classifies images using ResNet model.
 */
class ResNetClassifier : public vision_core::BaseModel {
public:
    ResNetClassifier(const std::string& model_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider",
                        const cv::Size& resize_size = cv::Size(256, 256),
                        const cv::Scalar& mean = cv::Scalar(0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f),
                        const cv::Scalar& std = cv::Scalar(0.229f * 255.0f, 0.224f * 255.0f, 0.225f * 255.0f),
                        bool center_crop = true,
                        int interpolation = cv::INTER_LINEAR);

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

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

protected:
    vision_core::InferResponse RunImpl(const vision_core::InferRequest& request) override;

private:
    vision_common::ClassificationResultList classify_input(
        const vision_core::ImageInput& input, int top_k = 5);
    vision_common::ClassificationResultList postprocess(
        std::vector<Ort::Value>& outputs, int top_k);

    int num_threads_;
    cv::Size resize_size_;
    cv::Scalar mean_;
    cv::Scalar std_;
    bool center_crop_;
    int interpolation_;
    std::string provider_;
    bool apply_softmax_ = true;
};

}  // namespace vision_deploy

#endif  // RESNET_CLASSIFIER_H
