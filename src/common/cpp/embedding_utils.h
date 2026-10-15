/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EMBEDDING_UTILS_H
#define EMBEDDING_UTILS_H

#include <vector>

namespace vision_common {

/**
 * @brief L2 normalize embedding vector
 * @param embedding Input embedding vector
 * @return Normalized embedding vector
 */
std::vector<float> normalize_embedding(const std::vector<float>& embedding);

/**
 * @brief Compute cosine similarity between two embeddings
 * @param embedding1 First embedding vector (normalization is not required)
 * @param embedding2 Second embedding vector (normalization is not required)
 * @return Similarity score (-1 to 1); 0 for zero-norm or non-finite inputs
 * @throws std::runtime_error if the vector dimensions differ
 */
float compute_similarity(const std::vector<float>& embedding1,
                        const std::vector<float>& embedding2);

}  // namespace vision_common

#endif  // EMBEDDING_UTILS_H

