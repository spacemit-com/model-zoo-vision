/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLOV8_DETECTOR_H
#define YOLOV8_DETECTOR_H

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
 * @brief YOLOv8 Object Detector
 * 
 * Supports detection with various input sizes and configurations.
 */
class YOLOv8Detector : public vision_core::BaseModel, public vision_core::IDetectionModel {
public:
    YOLOv8Detector(const std::string& model_path,
                    float conf_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    int num_threads = 4,
                    bool lazy_load = false,
                    const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~YOLOv8Detector() = default;

    /**
     * @brief Load YOLOv8 ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for YOLOv8 inference
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    std::vector<vision_common::Result> detect(const cv::Mat& image) override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    std::vector<vision_common::Result> postprocess(std::vector<Ort::Value>& outputs,
                                                    const cv::Size& orig_size);

private:
    float conf_threshold_;
    float iou_threshold_;
    int num_threads_;
    int num_classes_;
    std::string provider_;

    void get_dets(const cv::Size& orig_size, const float* boxes,
                    const float* scores, const float* score_sum,
                    const std::vector<int64_t>& dims,
                    int tensor_width, int tensor_height,
                    std::vector<vision_common::Result>& objects);
};

}  // namespace vision_deploy

#endif  // YOLOV8_DETECTOR_H
