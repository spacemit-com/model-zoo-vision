/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLO26_SEG_SEGMENTOR_H
#define YOLO26_SEG_SEGMENTOR_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class YOLO26SegSegmentor final : public vision_core::BaseModel {
public:
    YOLO26SegSegmentor(
        const std::string& model_path,
        float conf_threshold,
        float mask_threshold,
        int num_threads,
        bool lazy_load,
        const std::string& provider);

    void load_model() override;
    std::vector<vision_core::InferIntent> supported_intents()
        const override;
    std::vector<vision_core::ModelCapability> get_capabilities()
        const override;

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node& config, bool lazy_load);

protected:
    vision_core::InferResponse RunImpl(
        const vision_core::InferRequest& request) override;

private:
    cv::Mat preprocess(const cv::Mat& bgr) const;
    vision_common::SegmentationResultList segment_input(
        const vision_core::ImageInput& input,
        float conf_threshold,
        float mask_threshold,
        int max_det);
    vision_common::SegmentationResultList postprocess(
        const std::vector<Ort::Value>& outputs,
        const cv::Size& original_size,
        float conf_threshold,
        float mask_threshold,
        int max_det) const;

    float conf_threshold_;
    float mask_threshold_;
    int num_threads_;
    int input_width_ = 0;
    int input_height_ = 0;
    int proto_channels_ = 0;
    int proto_width_ = 0;
    int proto_height_ = 0;
    int64_t num_candidates_ = 0;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // YOLO26_SEG_SEGMENTOR_H
