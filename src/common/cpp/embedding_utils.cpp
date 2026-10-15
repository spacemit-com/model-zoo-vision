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
    // Compute L2 norm
    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm) + 1e-8f;

    // Normalize
    std::vector<float> normalized(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        normalized[i] = embedding[i] / norm;
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

