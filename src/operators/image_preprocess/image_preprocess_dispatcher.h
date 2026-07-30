/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMAGE_PREPROCESS_DISPATCHER_H
#define IMAGE_PREPROCESS_DISPATCHER_H

#include <memory>
#include <string>

#include "cpu_image_preprocessor.h"
#include "image_preprocess_backend.h"
#include "image_preprocessor.h"
#include "image_preprocess_result.h"
#include "image_preprocess_spec.h"
#include "core/cpp/vision_infer_types.h"

namespace vision_operators {

class ImagePreprocessDispatcher {
public:
    explicit ImagePreprocessDispatcher(
        PreprocessBackendPolicy policy =
            PreprocessBackendPolicy::kCpu);
    ImagePreprocessDispatcher(
        PreprocessBackendPolicy policy,
        ImagePreprocessorFactory opencl_factory);
    ~ImagePreprocessDispatcher();

    ImagePreprocessDispatcher(const ImagePreprocessDispatcher&) = delete;
    ImagePreprocessDispatcher& operator=(
        const ImagePreprocessDispatcher&) = delete;

    void configure(const std::string& backend);
    void reset();

    ImagePreprocessResult process(
        const vision_core::ImageInput& input,
        const ImagePreprocessSpec& spec,
        const CpuImagePreprocess& cpu_preprocess);

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace vision_operators

#endif  // IMAGE_PREPROCESS_DISPATCHER_H
