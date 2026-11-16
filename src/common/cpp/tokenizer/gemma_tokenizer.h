/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GEMMA_TOKENIZER_H
#define GEMMA_TOKENIZER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace vision_common {

class GemmaTokenizer final : public ITokenizer {
public:
    explicit GemmaTokenizer(const std::string& bin_path);

    TokenizedText Encode(const std::string& text, const TokenizerOptions& options = {}) override;

    static constexpr int kPadId = 0;
    static constexpr int kEosId = 1;

private:
    std::vector<int32_t> bpe_segment(const std::string& seg) const;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;
};

} // namespace vision_common

#endif // GEMMA_TOKENIZER_H
