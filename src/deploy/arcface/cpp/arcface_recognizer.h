/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARCFACE_RECOGNIZER_H
#define ARCFACE_RECOGNIZER_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief ArcFace Face Recognition Model
 * 
 * Extracts face embeddings for face recognition tasks.
 * The model should receive aligned face images as input.
 */
class ArcFaceRecognizer : public vision_core::BaseModel {
public:
    ArcFaceRecognizer(const std::string& model_path,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider",
                        float norm_std = 127.5f);

    virtual ~ArcFaceRecognizer() = default;

    /**
     * @brief Load ArcFace ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess face image for ArcFace inference
     * @param image Input face image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    std::vector<float> postprocess(std::vector<Ort::Value>& outputs);

protected:
    vision_core::InferResponse RunImpl(const vision_core::InferRequest& request) override;

private:
    vision_common::EmbeddingResult infer_embedding_input(
        const vision_core::ImageInput& input);

    int num_threads_;
    std::string provider_;
    float norm_std_ = 127.5f;
};

}  // namespace vision_deploy

#endif  // ARCFACE_RECOGNIZER_H
