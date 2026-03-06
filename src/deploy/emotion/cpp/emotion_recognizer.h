/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EMOTION_RECOGNIZER_H
#define EMOTION_RECOGNIZER_H

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
 * @brief Emotion Recognizer
 * 
 * Classifies emotions in face images.
 * Note: This model expects pre-cropped face images, not full images.
 */
class EmotionRecognizer : public vision_core::BaseModel, public vision_core::IClassificationModel {
public:
    EmotionRecognizer(const std::string& model_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~EmotionRecognizer() = default;

    /**
     * @brief Load Emotion ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess face image for emotion recognition
     * @param image Input face image in BGR format (should be cropped face)
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
    cv::Size target_size_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // EMOTION_RECOGNIZER_H
