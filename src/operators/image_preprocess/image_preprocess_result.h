/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMAGE_PREPROCESS_RESULT_H
#define IMAGE_PREPROCESS_RESULT_H

#include <functional>

#include <opencv2/core.hpp>

#include "image_preprocess_backend.h"

namespace vision_operators {

class ImagePreprocessResult {
public:
    ImagePreprocessResult(
        cv::Mat tensor,
        PreprocessBackend backend,
        std::function<void()> finish = {});
    ~ImagePreprocessResult();

    ImagePreprocessResult(const ImagePreprocessResult&) = delete;
    ImagePreprocessResult& operator=(const ImagePreprocessResult&) = delete;
    ImagePreprocessResult(ImagePreprocessResult&& other) noexcept;
    ImagePreprocessResult& operator=(
        ImagePreprocessResult&& other) noexcept;

    const cv::Mat& tensor() const noexcept;
    PreprocessBackend backend_used() const noexcept;
    void complete();

private:
    void finish() noexcept;

    cv::Mat tensor_;
    PreprocessBackend backend_{PreprocessBackend::kCpu};
    std::function<void()> finish_;
};

}  // namespace vision_operators

#endif  // IMAGE_PREPROCESS_RESULT_H
