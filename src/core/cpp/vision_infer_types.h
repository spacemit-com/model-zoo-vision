/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Internal inference request/response types (not installed to include/).
 */

#ifndef VISION_INFER_TYPES_H
#define VISION_INFER_TYPES_H

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <opencv2/core.hpp>

#include "common/cpp/datatype.h"

namespace vision_core {

enum class ImagePixelFormat {
    kBgr8,
    kNv12,
};

struct ImageInput {
    cv::Mat image;
    ImagePixelFormat format = ImagePixelFormat::kBgr8;
    int dma_fd = -1;
    bool has_initial_bbox = false;
    vision::BoundingBox initial_bbox;
};

struct StereoImageInput {
    ImageInput left;
    ImageInput right;
};

struct LocalFeaturePairInput {
    vision::LocalFeatures query;
    vision::LocalFeatures train;
};

struct SequenceInput {
    std::vector<float> pts;
    int image_width = 0;
    int image_height = 0;
};

struct TextInput {
    std::string text;
};

using InferInput = std::variant<
    ImageInput,
    StereoImageInput,
    LocalFeaturePairInput,
    SequenceInput,
    TextInput>;

enum class InferIntent {
    kUnspecified = -1,
    kDetect,
    kClassify,
    kEstimatePose,
    kSegment,
    kTrack,
    kEmbed,
    kEmbedText,
    kInferSequence,
    kOcr,
    kStereoDepth,
    kMonocularDepth,
    kExtractLocalFeatures,
    kMatchLocalFeatures,
};

// Unified inference-time parameters: nonpositive floats and negative integers
// use model defaults. Zero top_k/max_det requests no results.
struct InferParams {
    float conf_threshold = -1.f;
    float iou_threshold = -1.f;

    int top_k = -1;
    float kp_threshold = -1.f;
    float mask_threshold = -1.f;
    int max_det = -1;

    // Open-vocabulary text prompts (e.g. YOLO-World). Empty means "use the
    // model's configured default vocabulary". Ignored by models that don't
    // consume text.
    std::vector<std::string> prompts;
    std::vector<cv::Point2f> point_coords;
    std::vector<int> point_labels;
};

// For probability-valued overrides only. Nonpositive finite values preserve
// the model default; score domains such as RTMPose use their own validation.
inline float resolve_probability_threshold(
    float value, float default_value, const char* name) {
    if (!std::isfinite(value) || value > 1.0f) {
        throw std::invalid_argument(
            std::string(name) + " must be finite and <= 1 (<= 0 uses the model default)");
    }
    return value > 0.0f ? value : default_value;
}

struct InferRequest {
    InferInput input;
    InferIntent intent = InferIntent::kUnspecified;
    InferParams params;
};

enum class InferErrorCode {
    kInferenceFailed,
    kInvalidArgument,
};

struct InferResponse {
    std::vector<vision_common::ModelResult> results;
    bool ok = true;
    std::string error_message;
    // Only meaningful when !ok; existing model failures keep their status.
    InferErrorCode error_code = InferErrorCode::kInferenceFailed;
};

// Require explicit ownership transfer so a missed std::move cannot copy payloads.
template <typename Result>
InferResponse make_infer_response(std::vector<Result>&& results) {
    InferResponse response;
    response.results.reserve(results.size());
    for (auto& result : results) {
        response.results.emplace_back(std::move(result));
    }
    return response;
}

}  // namespace vision_core

#endif  // VISION_INFER_TYPES_H
