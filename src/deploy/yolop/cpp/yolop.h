/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLOP_H
#define YOLOP_H

#include <memory>
#include <string>
#include <vector>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

class YOLOP : public vision_core::BaseModel {
public:
    YOLOP(
        const std::string& model_path,
        float conf_threshold,
        float iou_threshold,
        int max_det,
        int num_threads,
        bool lazy_load,
        const std::string& provider);

    ~YOLOP() override = default;

    void load_model() override;

    std::vector<vision_core::InferIntent> supported_intents() const override;
    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node& config,
        bool lazy_load);

protected:
    vision_core::InferResponse RunImpl(
        const vision_core::InferRequest& request) override;

private:
    cv::Mat preprocess_cpu(const cv::Mat& image, vision_operators::ImagePreprocessGeometry* geometry) const;
    vision_core::InferResponse infer_input(
        const vision_core::ImageInput& input,
        float conf_threshold,
        float iou_threshold,
        int max_det);

    float conf_threshold_;
    float iou_threshold_;
    int max_det_;
    int num_threads_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // YOLOP_H
