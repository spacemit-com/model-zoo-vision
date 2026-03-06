/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VISION_MODEL_BASE_H
#define VISION_MODEL_BASE_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_order)

#include "spacemit_ort_env.h"

namespace vision_core {

enum class ModelCapability {
    kImageInput,
    kSequenceInput,
    kDraw,
    kEmbedding,
    kTrackUpdate
};

class BaseModel {
public:
    explicit BaseModel(const std::string& model_path, bool lazy_load = false);
    virtual ~BaseModel();

    virtual void load_model() = 0;

    virtual void warmup();

    virtual void release();

    virtual std::vector<ModelCapability> get_capabilities() const;

    bool supports_capability(ModelCapability capability) const;

    virtual std::vector<int64_t> get_input_shape() const;

    virtual std::string get_model_info() const;

protected:
    std::string model_path_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    std::vector<int64_t> input_shape_;
    bool model_loaded_;
    bool lazy_load_;

    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    size_t output_num_ = 0;
    Ort::AllocatorWithDefaultOptions allocator_;

    void ensure_model_loaded();

    void init_session(int num_threads = 4, const std::string& provider = "SpaceMITExecutionProvider");

    std::vector<Ort::Value> run_session(const cv::Mat& input_blob);

    Ort::MemoryInfo memory_info_{nullptr};
};

}  // namespace vision_core

#endif  // VISION_MODEL_BASE_H
