/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
// CLIP BPE tokenizer, shared by CLIP/MobileCLIP text encoder deployments.
//
// Reads bpe_merges.txt (the 48894 merges CLIP uses, one "a b" pair per line).
// Known limitation: ftfy.fix_text() is not ported (mojibake repair / NFC /
// ligature folding); plain class names are not affected. html.unescape (x2),
// whitespace cleanup, lowercasing and pre-tokenization follow Python behaviour
// in kClip mode. kLegacy preserves the older deployed tokenizer instead.
#ifndef CLIP_TOKENIZER_H
#define CLIP_TOKENIZER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace vision_common {

enum class ClipTokenizerMode {
    kClip,
    // Preserve existing YOLO-World/YOLOE/MobileCLIP2 whitespace pre-tokenization
    // and byte-wise BPE behavior. Not interchangeable with kClip for Unicode.
    kLegacy,
};

class ClipTokenizer final : public ITokenizer {
public:
    explicit ClipTokenizer(const std::string& merges_path, ClipTokenizerMode mode = ClipTokenizerMode::kClip);

    TokenizedText Encode(const std::string& text, const TokenizerOptions& options = {}) override;

    static constexpr int32_t kSot = 49406;
    static constexpr int32_t kEot = 49407;

private:
    std::string bpe(const std::string& token) const;
    std::vector<int64_t> encode_body(const std::string& text) const;

    ClipTokenizerMode mode_;
    std::vector<std::string> byte_encoder_; // byte -> UTF-8 of its mapped code point
    std::unordered_map<std::string, int32_t> encoder_;
    std::unordered_map<std::string, int> bpe_ranks_; // key: first + '\x01' + second
    mutable std::unordered_map<std::string, std::string> cache_;
};

} // namespace vision_common

#endif // CLIP_TOKENIZER_H
