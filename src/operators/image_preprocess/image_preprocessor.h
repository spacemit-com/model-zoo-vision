/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMAGE_PREPROCESSOR_H
#define IMAGE_PREPROCESSOR_H

#include <functional>
#include <memory>
#include <stdexcept>

#include <opencv2/core.hpp>

#include "image_preprocess_spec.h"
#include "core/cpp/vision_infer_types.h"

namespace vision_operators {

class ImagePreprocessBackendUnavailable
    : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ImagePreprocessor {
public:
    virtual ~ImagePreprocessor() = default;

    virtual cv::Mat process(
        const vision_core::ImageInput& input) = 0;
    virtual void complete() = 0;
};

using ImagePreprocessorFactory = std::function<
    std::shared_ptr<ImagePreprocessor>(
        const ImagePreprocessSpec&)>;

std::shared_ptr<ImagePreprocessor>
create_opencl_image_preprocessor(
    const ImagePreprocessSpec& spec,
    int output_ring_depth = 3);

bool opencl_image_preprocessor_compiled() noexcept;

}  // namespace vision_operators

#endif  // IMAGE_PREPROCESSOR_H
