/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef COSPLACE_RECOGNIZER_H
#define COSPLACE_RECOGNIZER_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "vision_task_interfaces.h"
#include "embedding_utils.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

/**
 * @brief CosPlace Visual Place Recognition Model
 * 
 * Extracts global descriptors for visual place recognition tasks.
 * The model encodes images into compact feature vectors for retrieval.
 */
class CosPlaceRecognizer : public vision_core::BaseModel, public vision_core::IEmbeddingModel {
public:
    CosPlaceRecognizer(const std::string& model_path,
            int num_threads = 4,
            bool lazy_load = false,
            const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~CosPlaceRecognizer() = default;

    /**
     * @brief Load CosPlace ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for CosPlace inference
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    /**
     * @brief Run inference and return place embedding
     * @param image Input image in BGR format
     * @return Place embedding result with normalized descriptor
     */
    vision_common::EmbeddingResult infer_embedding(const cv::Mat& image) override;

    vision_core::InferResponse Run(const vision_core::InferRequest& request) override;

    std::vector<vision_core::InferIntent> supported_intents() const override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /**
     * @brief Compute L2 distance between two place embeddings
     * @param embedding1 First place embedding
     * @param embedding2 Second place embedding
     * @return L2 distance (lower means more similar)
     */
    static float compute_distance(const std::vector<float>& embedding1,
            const std::vector<float>& embedding2);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    std::vector<float> postprocess(std::vector<Ort::Value>& outputs);

private:
    vision_common::EmbeddingResult infer_embedding_input(
        const vision_core::ImageInput& input);

    int num_threads_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // COSPLACE_RECOGNIZER_H
