/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <optional>
#include <string>

#include "tokenizer/tokenizer_types.h"

namespace vision_common {

// Internal model-independent interface. Instances may own mutable caches;
// callers serialize access to one instance. No ONNX/session dependencies.
// Default options return an unlimited, unpadded sequence with special tokens.
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual TokenizedText Encode(const std::string& text, const TokenizerOptions& options = {}) = 0;

protected:
    struct SpecialTokens {
        std::optional<int64_t> bos;
        std::optional<int64_t> eos;
        int64_t pad = 0;
    };

    // Truncates the body before adding special tokens, preserving BOS/EOS.
    // Invalid options and overlong inputs without truncation throw invalid_argument.
    static TokenizedText Finalize(
        std::vector<int64_t> ids, const TokenizerOptions& options, const SpecialTokens& special);
};

} // namespace vision_common

#endif // TOKENIZER_H
