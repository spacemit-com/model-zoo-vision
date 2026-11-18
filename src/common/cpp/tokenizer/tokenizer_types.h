/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TOKENIZER_TYPES_H
#define TOKENIZER_TYPES_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vision_common {

enum class TokenizerPadding {
    kNone,
    kMaxLength,
};

struct TokenizerOptions {
    // Includes special tokens; zero means unlimited. Fixed padding requires > 0.
    size_t max_length = 0;
    bool truncation = false;
    bool add_special_tokens = true;
    TokenizerPadding padding = TokenizerPadding::kNone;
};

struct TokenizedText {
    std::vector<int64_t> input_ids;
    // Same length as input_ids: 1 for real/special tokens, 0 for padding.
    std::vector<int64_t> attention_mask;
    // Empty when the tokenizer does not produce segment IDs.
    std::vector<int64_t> token_type_ids;
    bool truncated = false;
};

} // namespace vision_common

#endif // TOKENIZER_TYPES_H
