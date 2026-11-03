/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "embedding_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vision_common {

std::vector<float> normalize_embedding(const std::vector<float>& embedding) {
    // Double accumulation is safe for squares of finite float32 values.
    double norm_squared = 0.0;
    for (float val : embedding) {
        if (!std::isfinite(val)) {
            throw std::runtime_error("Embedding values must be finite");
        }
        const double value = val;
        norm_squared += value * value;
    }
    const double norm = std::sqrt(norm_squared);

    // Normalize
    std::vector<float> normalized(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        normalized[i] = norm > 0.0
            ? static_cast<float>(embedding[i] / norm) : 0.0f;
    }

    return normalized;
}

float compute_similarity(const std::vector<float>& embedding1,
                        const std::vector<float>& embedding2) {
    if (embedding1.size() != embedding2.size()) {
        throw std::runtime_error("Embedding dimensions do not match");
    }

    // Double accumulation avoids overflow/underflow for finite float inputs.
    double dot = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    for (size_t i = 0; i < embedding1.size(); ++i) {
        const double a = embedding1[i];
        const double b = embedding2[i];
        if (!std::isfinite(a) || !std::isfinite(b)) return 0.0f;
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0f;
    return std::clamp(
        static_cast<float>(dot / (std::sqrt(norm1) * std::sqrt(norm2))),
        -1.0f, 1.0f);
}

}  // namespace vision_common

