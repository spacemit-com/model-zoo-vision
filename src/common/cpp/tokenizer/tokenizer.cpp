/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tokenizer/tokenizer.h"

#include <stdexcept>
#include <utility>

namespace vision_common {

TokenizedText ITokenizer::Finalize(
    std::vector<int64_t> ids, const TokenizerOptions& options, const SpecialTokens& special) {
    if (options.padding != TokenizerPadding::kNone && options.padding != TokenizerPadding::kMaxLength) {
        throw std::invalid_argument("Invalid tokenizer padding mode");
    }
    if (options.padding == TokenizerPadding::kMaxLength && options.max_length == 0) {
        throw std::invalid_argument("Tokenizer fixed padding requires max_length > 0");
    }
    const size_t special_count = options.add_special_tokens
        ? static_cast<size_t>(special.bos.has_value()) + static_cast<size_t>(special.eos.has_value())
        : 0;
    if (options.max_length > 0 && options.max_length < special_count) {
        throw std::invalid_argument("Tokenizer max_length cannot hold required special tokens");
    }
    TokenizedText result;
    if (options.max_length > 0 && ids.size() > options.max_length - special_count) {
        if (!options.truncation) {
            throw std::invalid_argument("Text exceeds tokenizer max_length with truncation disabled");
        }
        ids.resize(options.max_length - special_count);
        result.truncated = true;
    }
    if (options.add_special_tokens) {
        if (special.bos) {
            ids.insert(ids.begin(), *special.bos);
        }
        if (special.eos) {
            ids.push_back(*special.eos);
        }
    }
    result.attention_mask.assign(ids.size(), 1);
    if (options.padding == TokenizerPadding::kMaxLength) {
        ids.resize(options.max_length, special.pad);
        result.attention_mask.resize(options.max_length, 0);
    }
    result.input_ids = std::move(ids);
    return result;
}

} // namespace vision_common
