/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef BANET2D_H
#define BANET2D_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vision_model_base.h"

namespace YAML {
class Node;
}

namespace vision_deploy {

struct BANetLetterbox {
    int input_width = 0;
    int input_height = 0;
    int output_width = 0;
    int output_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int pad_left = 0;
    int pad_top = 0;
    int pad_right = 0;
    int pad_bottom = 0;
};

BANetLetterbox make_banet_letterbox(
    int input_width,
    int input_height,
    int output_width,
    int output_height);

cv::Mat restore_banet_disparity(
    const cv::Mat& model_disparity,
    const BANetLetterbox& geometry,
    const cv::Size& original_size);

class BANet2D final : public vision_core::BaseModel {
public:
    BANet2D(
        const std::string& model_path,
        int num_threads,
        bool lazy_load,
        const std::string& provider);

    void load_model() override;
    vision_core::InferResponse Run(
        const vision_core::InferRequest& request) override;
    std::vector<vision_core::InferIntent> supported_intents() const override;
    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    static std::unique_ptr<vision_core::BaseModel> create(
        const YAML::Node& config,
        bool lazy_load);

private:
    cv::Mat preprocess_one(
        const cv::Mat& bgr,
        const BANetLetterbox& geometry) const;
    vision::Disparity infer_stereo(
        const vision_core::StereoImageInput& input);

    int num_threads_;
    std::string provider_;
};

}  // namespace vision_deploy

#endif  // BANET2D_H
