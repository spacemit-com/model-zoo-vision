/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NANOTRACK_TRACKER_H
#define NANOTRACK_TRACKER_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_order)

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

struct NanoTrackParams {
    int num_threads = 4;
    int template_num_threads = 4;
    int head_num_threads = 1;
    float context_amount = 0.5f;
    float penalty_k = 0.148f;
    float window_influence = 0.462f;
    float learning_rate = 0.390f;
};

std::vector<cv::Point2f> generate_nanotrack_points(
    int stride,
    int score_size);

std::vector<float> nanotrack_foreground_probabilities(
    const float* logits,
    int score_size);

class NanoTracker final : public vision_core::BaseModel {
public:
    NanoTracker(
        const std::string& search_model_path,
        std::string template_model_path,
        std::string head_model_path,
        NanoTrackParams params,
        bool lazy_load,
        std::string provider);

    void load_model() override;
    void release() override;
    vision_core::InferResponse Run(
        const vision_core::InferRequest& request) override;
    std::vector<vision_core::InferIntent> supported_intents() const override;
    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node& config,
        bool lazy_load);

private:
    vision::Tracking initialize(
        const cv::Mat& image,
        const vision::BoundingBox& initial_box);
    vision::Tracking track(const cv::Mat& image);
    cv::Mat preprocess_crop(
        const cv::Mat& image,
        const cv::Point2f& center,
        int output_size,
        int crop_size) const;
    void initialize_auxiliary_sessions();

    static constexpr int kTemplateSize = 127;
    static constexpr int kSearchSize = 255;
    static constexpr int kScoreSize = 16;
    static constexpr int kPointStride = 16;
    static constexpr float kMinimumBoxSize = 10.0f;

    std::string template_model_path_;
    std::string head_model_path_;
    NanoTrackParams params_;
    std::string provider_;

    std::unique_ptr<Ort::Session> template_session_;
    std::unique_ptr<Ort::Session> head_session_;
    std::vector<std::string> template_input_names_storage_;
    std::vector<std::string> template_output_names_storage_;
    std::vector<const char*> template_input_names_;
    std::vector<const char*> template_output_names_;
    std::vector<std::string> head_input_names_storage_;
    std::vector<std::string> head_output_names_storage_;
    std::vector<const char*> head_input_names_;
    std::vector<const char*> head_output_names_;

    bool initialized_ = false;
    cv::Rect2f state_box_;
    cv::Point2f state_center_;
    cv::Scalar channel_average_;
    std::vector<float> template_features_;
    std::vector<int64_t> template_feature_shape_;
    std::vector<cv::Point2f> points_;
    std::vector<float> window_;
};

}  // namespace vision_deploy

#endif  // NANOTRACK_TRACKER_H
