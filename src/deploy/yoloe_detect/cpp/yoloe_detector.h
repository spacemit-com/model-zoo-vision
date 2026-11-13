/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef YOLOE_DETECTOR_H
#define YOLOE_DETECTOR_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"
#include "yoloe_clip_tokenizer.hpp"

namespace YAML {
class Node;
}
namespace vision_deploy {

std::vector<std::string> yoloe_prompt_names(const std::vector<std::string>& texts,
    const std::vector<vision::VisualPrompt>& visuals, const std::vector<std::string>& defaults);
std::vector<float> yoloe_visual_mask(const vision::VisualPrompt& prompt, int side);
vision_common::DetectionResultList decode_yoloe_detections(const float* data, size_t count, int slots, int anchors,
    int classes, const vision_operators::ImagePreprocessGeometry& geometry, float confidence, int max_det);

class YOLOEDetector final : public vision_core::BaseModel {
public:
    YOLOEDetector(const YAML::Node& config, bool lazy_load);
    void load_model() override;
    void release() override;
    std::vector<vision_core::InferIntent> supported_intents() const override;
    std::vector<vision_core::ModelCapability> get_capabilities() const override;
    std::vector<std::string> get_dynamic_class_names() const override;
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

protected:
    vision_core::InferResponse RunImpl(const vision_core::InferRequest& request) override;

private:
    void ensure_prompt(const std::vector<std::string>& texts, const std::vector<vision::VisualPrompt>& visuals);
    std::vector<float> encode_text(const std::vector<std::string>& names);
    std::vector<float> encode_visual(const vision::VisualPrompt& prompt);
    void open_text();
    void open_visual();
    std::vector<float> run_encoder(Ort::Session& session, std::vector<Ort::Value>& inputs,
        const std::vector<int64_t>& shape, const char* component);

    std::string text_path_, vpe_path_, merges_path_, provider_;
    int threads_ = 8, prompt_threads_ = 2, reference_side_ = 0;
    float confidence_ = 0.2F;
    std::unique_ptr<Ort::Session> text_session_, vpe_session_;
    std::unique_ptr<ClipTokenizer> tokenizer_;
    std::vector<std::string> defaults_, labels_;
    std::vector<vision::VisualPrompt> cached_visuals_;
    std::vector<float> prompt_data_;
    bool cached_text_ = false, prompt_ready_ = false;
};
} // namespace vision_deploy
#endif // YOLOE_DETECTOR_H
