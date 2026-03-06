/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef STGCN_ACTION_RECOGNIZER_H
#define STGCN_ACTION_RECOGNIZER_H

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
 * @brief STGCN / TSSTG Action Recognizer (C++)
 *
 * Same logic as Python deploy: 30-frame skeleton sequence -> 7-class action probabilities.
 * Input: pts (T=30, V=13, C=3) in pixel coordinates; image_size for normalization.
 * Output: 7 class probs, index 6 = Fall Down.
 */
class StgcnActionRecognizer : public vision_core::BaseModel, public vision_core::ISequenceActionModel {
public:
    static constexpr int kSequenceLength = 30;
    static constexpr int kNumKeypoints = 13;
    static constexpr int kFallDownClassIndex = 6;
    static constexpr int kNumClasses = 7;

    StgcnActionRecognizer(const std::string& model_path,
                            int num_threads = 4,
                            bool lazy_load = false,
                            const std::string& provider = "CPUExecutionProvider");

    virtual ~StgcnActionRecognizer() = default;

    void load_model() override;

    /**
     * @brief Run action recognition on skeleton sequence (same as Python predict).
     * @param pts Flat array of size kSequenceLength * kNumKeypoints * 3, layout [t][v][c] (x, y, score) in pixel coords
     * @param image_width  Image width for normalizing coordinates
     * @param image_height Image height for normalizing coordinates
     * @return Vector of 7 class probabilities (index 6 = Fall Down)
     */
    std::vector<float> predict(const float* pts,
                                int image_width,
                                int image_height);

    std::vector<float> infer_sequence(const float* pts,
                                        int image_width,
                                        int image_height) override;

    std::vector<vision_core::ModelCapability> get_capabilities() const override;

    /** Overload: pts as vector. */
    std::vector<float> predict(const std::vector<float>& pts,
                                int image_width,
                                int image_height);

    std::vector<std::string> get_class_names() const override { return class_names_; }
    int get_fall_down_class_index() const override { return kFallDownClassIndex; }
    std::string get_class_name(int pred_class) const;

    static std::unique_ptr<vision_core::BaseModel> create(const YAML::Node& config, bool lazy_load);

private:
    int num_threads_;
    std::string provider_;
    std::vector<std::string> class_names_;

    /** Run ONNX with two inputs (pts_tensor, mot). */
    std::vector<Ort::Value> run_session_two_inputs(const float* pts_tensor_data,
                                                    const std::vector<int64_t>& pts_shape,
                                                    const float* mot_data,
                                                    const std::vector<int64_t>& mot_shape);
};

}  // namespace vision_deploy

#endif  // STGCN_ACTION_RECOGNIZER_H
