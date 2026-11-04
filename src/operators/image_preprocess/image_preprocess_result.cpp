/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_preprocess_result.h"

#include <utility>

namespace vision_operators {

ImagePreprocessResult::ImagePreprocessResult(
    cv::Mat tensor,
    PreprocessBackend backend,
    const ImagePreprocessGeometry& geometry,
    std::function<void()> finish)
    : tensor_(std::move(tensor)),
        backend_(backend),
        geometry_(geometry),
        finish_(std::move(finish))
{
}

ImagePreprocessResult::~ImagePreprocessResult()
{
    finish();
}

ImagePreprocessResult::ImagePreprocessResult(
    ImagePreprocessResult&& other) noexcept
    : tensor_(std::move(other.tensor_)),
        backend_(other.backend_),
        geometry_(other.geometry_),
        finish_(std::move(other.finish_))
{
}

ImagePreprocessResult& ImagePreprocessResult::operator=(
    ImagePreprocessResult&& other) noexcept
{
    if (this != &other) {
        finish();
        tensor_ = std::move(other.tensor_);
        backend_ = other.backend_;
        geometry_ = other.geometry_;
        finish_ = std::move(other.finish_);
    }
    return *this;
}

const cv::Mat& ImagePreprocessResult::tensor() const noexcept
{
    return tensor_;
}

PreprocessBackend
ImagePreprocessResult::backend_used() const noexcept
{
    return backend_;
}

const ImagePreprocessGeometry& ImagePreprocessResult::geometry() const
{
    geometry_.validate();
    return geometry_;
}

void ImagePreprocessResult::complete()
{
    if (!finish_) return;
    std::function<void()> completion = std::move(finish_);
    completion();
}

void ImagePreprocessResult::finish() noexcept
{
    if (!finish_) return;
    try {
        finish_();
    } catch (...) {
    }
    finish_ = {};
}

}  // namespace vision_operators
