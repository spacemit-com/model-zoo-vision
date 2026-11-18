/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tokenizer/gemma_tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace vision_common {

namespace {

void read_exact(std::ifstream& f, void* data, size_t size, uint64_t& remaining) {
    if (size > remaining || !f.read(static_cast<char*>(data), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("GemmaTokenizer: truncated or unreadable tokenizer data");
    }
    remaining -= size;
}

template <typename T> T read_value(std::ifstream& f, uint64_t& remaining) {
    T value = 0;
    read_exact(f, &value, sizeof(value), remaining);
    return value;
}

std::string read_str(std::ifstream& f, uint64_t& remaining) {
    const auto len = read_value<uint16_t>(f, remaining);
    if (len > remaining) {
        throw std::runtime_error("GemmaTokenizer: token length exceeds remaining file data");
    }
    std::string s(len, '\0');
    if (len > 0) {
        read_exact(f, s.data(), len, remaining);
    }
    return s;
}

std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int len = 1;
        if ((c & 0x80) == 0x00) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        }
        chars.push_back(s.substr(i, static_cast<size_t>(len)));
        i += static_cast<size_t>(len);
    }
    return chars;
}

} // namespace

GemmaTokenizer::GemmaTokenizer(const std::string& bin_path) {
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("GemmaTokenizer: cannot open tokenizer: " + bin_path);
    }

    const auto file_size = f.tellg();
    if (file_size < std::streampos(0) || !f.seekg(0, std::ios::beg)) {
        throw std::runtime_error("GemmaTokenizer: cannot determine tokenizer size: " + bin_path);
    }
    auto remaining = static_cast<uint64_t>(file_size);
    const auto vocab_size = read_value<uint32_t>(f, remaining);
    // IDs/ranks are int32_t. Each entry needs a length prefix, and the
    // vocabulary must leave room for merge_count and contain PAD/EOS IDs.
    if (vocab_size <= kEosId || vocab_size > INT32_MAX || remaining < sizeof(uint32_t) ||
        vocab_size > (remaining - sizeof(uint32_t)) / sizeof(uint16_t)) {
        throw std::runtime_error("GemmaTokenizer: invalid vocabulary count: " + bin_path);
    }
    id_to_token_.resize(vocab_size);
    token_to_id_.reserve(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        id_to_token_[i] = read_str(f, remaining);
        token_to_id_[id_to_token_[i]] = static_cast<int32_t>(i);
    }

    const auto merge_count = read_value<uint32_t>(f, remaining);
    if (merge_count > INT32_MAX || merge_count > remaining / (2 * sizeof(uint16_t))) {
        throw std::runtime_error("GemmaTokenizer: invalid merge count: " + bin_path);
    }
    merge_rank_.reserve(merge_count);
    for (uint32_t rank = 0; rank < merge_count; ++rank) {
        const std::string a = read_str(f, remaining);
        const std::string b = read_str(f, remaining);
        merge_rank_[a + '\x00' + b] = static_cast<int32_t>(rank);
    }
    if (remaining != 0) {
        throw std::runtime_error("GemmaTokenizer: unexpected trailing tokenizer data: " + bin_path);
    }
}

std::vector<int32_t> GemmaTokenizer::bpe_segment(const std::string& seg) const {
    auto chars = split_utf8(seg);
    if (chars.empty()) {
        return {};
    }

    std::vector<std::string> tokens = chars;
    while (tokens.size() > 1) {
        int best_rank = INT32_MAX;
        int best_i = -1;
        for (int i = 0; i < static_cast<int>(tokens.size()) - 1; ++i) {
            const std::string key = tokens[static_cast<size_t>(i)] + '\x00' + tokens[static_cast<size_t>(i) + 1];
            const auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_i == -1) {
            break;
        }
        tokens[static_cast<size_t>(best_i)] += tokens[static_cast<size_t>(best_i) + 1];
        tokens.erase(tokens.begin() + best_i + 1);
    }

    std::vector<int32_t> ids;
    ids.reserve(tokens.size());
    for (const auto& tok : tokens) {
        const auto it = token_to_id_.find(tok);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
            continue;
        }
        for (unsigned char byte : tok) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", byte);
            const auto it2 = token_to_id_.find(buf);
            if (it2 != token_to_id_.end()) {
                ids.push_back(it2->second);
            }
        }
    }
    return ids;
}

TokenizedText GemmaTokenizer::Encode(const std::string& text, const TokenizerOptions& options) {
    static const std::string kSpaceMark = "\xe2\x96\x81";
    std::string normalized;
    normalized.reserve(text.size() + 32);
    for (char ch : text) {
        if (ch == ' ') {
            normalized += kSpaceMark;
        } else {
            normalized += ch;
        }
    }

    const auto ids32 = bpe_segment(normalized);
    return Finalize(std::vector<int64_t>(ids32.begin(), ids32.end()), options, {std::nullopt, kEosId, kPadId});
}

} // namespace vision_common
