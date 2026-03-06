/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef YOLOV8_SEG_DETECTOR_H
#define YOLOV8_SEG_DETECTOR_H

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
 * @brief YOLOv8-Seg Instance Segmentation Detector
 * 
 * Detects objects and generates segmentation masks using YOLOv8-Seg model.
 */
class YOLOv8SegDetector : public vision_core::BaseModel, public vision_core::ISegmentationModel {
public:
    YOLOv8SegDetector(const std::string& model_path,
                        float conf_threshold = 0.25f,
                        float iou_threshold = 0.45f,
                        int num_threads = 4,
                        bool lazy_load = false,
                        const std::string& provider = "SpaceMITExecutionProvider");

    virtual ~YOLOv8SegDetector() = default;

    /**
     * @brief Load YOLOv8-Seg ONNX model
     */
    void load_model() override;

    /**
     * @brief Preprocess image for YOLOv8-Seg inference
     * @param image Input image in BGR format
     * @return Preprocessed tensor
     */
    cv::Mat preprocess(const cv::Mat& image);

    std::vector<vision_common::Result> segment(const cv::Mat& image) override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    // Factory hook: used by vision_core::ModelRegistrar for self-registration
    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

    /** @brief Postprocess (task layer, callable separately e.g. for benchmark). */
    std::vector<vision_common::Result> postprocess(std::vector<Ort::Value>& outputs, const cv::Size& orig_size);

private:
    float conf_threshold_;
    float iou_threshold_;
    int num_threads_;
    int num_classes_;
    int proto_channels_;
    std::string provider_;

    void Get_Dets(const cv::Size& orig_size, const float* boxes, const float* scores,
                    const float* score_sum, const float* seg_part, std::vector<int64_t> dims,
                    int tensor_width, int tensor_height, int num_classes,
                    std::vector<vision_common::Result>& objects);
    std::vector<std::shared_ptr<cv::Mat>> _process_masks(
        const float* protos,
        const std::vector<int64_t>& proto_dims,
        const std::vector<std::vector<float>>& mask_coeffs,
        const std::vector<vision_common::Result>& results,
        const cv::Size& orig_shape);
};

}  // namespace vision_deploy

#endif  // YOLOV8_SEG_DETECTOR_H
