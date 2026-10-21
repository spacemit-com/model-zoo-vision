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

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief Emotion Recognizer
 *
 * Classifies emotions in face images.
 * Note: This model expects pre-cropped face images, not full images.
 *
 * Feature mode: when constructed with feature_mode=true, the model acts as a
 * feature extractor (kEmbed) instead of a classifier (kClassify). Inference
 * then returns the raw output vector (e.g. 512-d) without argmax or L2 norm.
 * This is used as the backbone for dynamic (LSTM) emotion recognition.
 */
class EmotionRecognizer : public vision_core::BaseModel {
public:
    EmotionRecognizer(const std::string& model_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider",
                        bool feature_mode = false);

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

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

protected:
    vision_core::InferResponse RunImpl(const vision_core::InferRequest& request) override;

private:
    PreparedImage prepare_input(
        const vision_core::ImageInput& input);
    vision_common::ClassificationResultList classify_input(
        const vision_core::ImageInput& input, int top_k = 1);
    vision_common::ClassificationResultList postprocess(
        std::vector<Ort::Value>& outputs, int top_k);
    vision_common::EmbeddingResult infer_embedding_input(
        const vision_core::ImageInput& input);

    int num_threads_;
    cv::Size target_size_;
    std::string provider_;
    bool feature_mode_;
};

}  // namespace vision_deploy

#endif  // EMOTION_RECOGNIZER_H
