/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
// YOLOE CLIP BPE tokenizer, a C++ port of clip/simple_tokenizer.py +
// clip.tokenize(truncate=True).
//
// Reads bpe_merges.txt (the 48894 merges CLIP uses, one "a b" pair per line).
// Known limitation: ftfy.fix_text() is not ported (mojibake repair / NFC /
// ligature folding); plain class names are not affected. html.unescape (x2),
// whitespace cleanup, lowercasing and the pre-tokenization regex follow the
// Python behaviour exactly.
#ifndef YOLOE_CLIP_TOKENIZER_HPP
#define YOLOE_CLIP_TOKENIZER_HPP

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class ClipTokenizer {
public:
    explicit ClipTokenizer(const std::string& merges_path);

    // Same as clip.tokenize([text], context_length, truncate=True)[0].
    std::vector<int32_t> tokenize(const std::string& text, int context_length = 77) const;

    // Same as SimpleTokenizer.encode(text): BPE ids without SOT/EOT/padding.
    std::vector<int32_t> encode(const std::string& text) const;

    static constexpr int32_t kSot = 49406;
    static constexpr int32_t kEot = 49407;

private:
    std::string bpe(const std::string& token) const;

    std::vector<std::string> byte_encoder_; // byte -> UTF-8 of its mapped code point
    std::unordered_map<std::string, int32_t> encoder_;
    std::unordered_map<std::string, int> bpe_ranks_; // key: first + '\x01' + second
    mutable std::unordered_map<std::string, std::string> cache_;
};

#endif // YOLOE_CLIP_TOKENIZER_HPP
