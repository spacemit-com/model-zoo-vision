/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENCL_IMAGE_PREPROCESSOR_H
#define OPENCL_IMAGE_PREPROCESSOR_H

#include <memory>

#include <opencv2/core.hpp>

#include "operators/image_preprocess/image_preprocessor.h"
#include "operators/image_preprocess/image_preprocess_spec.h"
#include "vision_infer_types.h"

namespace vision_opencl {

class OpenClImagePreprocessor final
    : public vision_operators::ImagePreprocessor {
public:
    explicit OpenClImagePreprocessor(
        const vision_operators::ImagePreprocessSpec& spec,
        int output_ring_depth = 3);
    ~OpenClImagePreprocessor();

    OpenClImagePreprocessor(const OpenClImagePreprocessor&) = delete;
    OpenClImagePreprocessor& operator=(const OpenClImagePreprocessor&) = delete;

    cv::Mat process(
        const vision_core::ImageInput& input) override;
    void complete() override;
    void finish_cpu_read();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vision_opencl

#endif  // OPENCL_IMAGE_PREPROCESSOR_H
