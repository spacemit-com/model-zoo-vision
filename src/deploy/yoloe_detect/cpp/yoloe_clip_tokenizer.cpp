/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
// C++ port of clip/simple_tokenizer.py. Every step mirrors the Python source;
// see yoloe_clip_tokenizer.hpp.
#include "yoloe_clip_tokenizer.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "html_entities.hpp"
#include "unicode_tables.hpp"

namespace {

using U32 = std::u32string;

// ---------------------------------------------------------------- UTF-8 <->
// code points
U32 utf8_decode(const std::string& s) {
    U32 out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        char32_t cp;
        int n;
        if (c < 0x80) {
            cp = c;
            n = 1;
        } else if ((c >> 5) == 0x6) {
            cp = c & 0x1F;
            n = 2;
        } else if ((c >> 4) == 0xE) {
            cp = c & 0x0F;
            n = 3;
        } else if ((c >> 3) == 0x1E) {
            cp = c & 0x07;
            n = 4;
        } else {
            throw std::runtime_error("invalid UTF-8 input");
        }
        if (i + n > s.size())
            throw std::runtime_error("truncated UTF-8 input");
        for (int k = 1; k < n; ++k) {
            unsigned char cc = s[i + k];
            if ((cc >> 6) != 0x2)
                throw std::runtime_error("invalid UTF-8 input");
            cp = (cp << 6) | (cc & 0x3F);
        }
        out.push_back(cp);
        i += n;
    }
    return out;
}

void utf8_append(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

std::string utf8_encode(const U32& s) {
    std::string out;
    for (char32_t c : s)
        utf8_append(out, c);
    return out;
}

// ---------------------------------------------------------------- Unicode
// classes (tables from Python)
template <size_t N> bool in_ranges(const unicode_tables::CodeRange (&r)[N], char32_t c) {
    const auto* it = std::upper_bound(
        r, r + N, static_cast<uint32_t>(c), [](uint32_t v, const unicode_tables::CodeRange& x) { return v < x.lo; });
    return it != r && c <= (it - 1)->hi;
}
bool is_letter(char32_t c) { return in_ranges(unicode_tables::kLetter, c); } // regex \p{L}
bool is_number(char32_t c) { return in_ranges(unicode_tables::kNumber, c); } // regex \p{N}
bool is_re_space(char32_t c) { return in_ranges(unicode_tables::kSpace, c); } // regex \s
bool is_py_space(char32_t c) { return in_ranges(unicode_tables::kPySpace, c); } // str.isspace / str.strip

bool is_cased(char32_t c) { return in_ranges(unicode_tables::kCased, c); }
bool is_case_ignorable(char32_t c) { return in_ranges(unicode_tables::kCaseIgnorable, c); }

// CPython handle_capital_sigma: U+03A3 lowers to final sigma U+03C2 when
// preceded by a cased letter and not followed by one (skipping case-ignorable
// characters in both directions).
bool final_sigma(const U32& s, size_t i) {
    size_t j = i;
    while (j > 0 && is_case_ignorable(s[j - 1]))
        --j;
    if (j == 0 || !is_cased(s[j - 1]))
        return false;
    size_t k = i + 1;
    while (k < s.size() && is_case_ignorable(s[k]))
        ++k;
    return k == s.size() || !is_cased(s[k]);
}

U32 py_lower(const U32& s) { // str.lower()
    static const std::unordered_map<char32_t, const unicode_tables::LowerMap*> table = [] {
        std::unordered_map<char32_t, const unicode_tables::LowerMap*> m;
        for (const auto& e : unicode_tables::kLower)
            m[e.cp] = &e;
        return m;
    }();
    U32 out;
    for (size_t i = 0; i < s.size(); ++i) {
        char32_t c = s[i];
        if (c == 0x3A3) {
            out.push_back(final_sigma(s, i) ? 0x3C2 : 0x3C3);
            continue;
        }
        auto it = table.find(c);
        if (it == table.end())
            out.push_back(c);
        else
            out.append(it->second->to, it->second->to + it->second->n);
    }
    return out;
}

U32 py_strip(const U32& s) {
    size_t b = 0, e = s.size();
    while (b < e && is_py_space(s[b]))
        ++b;
    while (e > b && is_py_space(s[e - 1]))
        --e;
    return s.substr(b, e - b);
}

// ----------------------------------------------------------------
// html.unescape (Python 3.8 stdlib)
bool is_ascii_digit(char32_t c) { return c >= '0' && c <= '9'; }
bool is_ascii_hex(char32_t c) { return is_ascii_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

std::string replace_charref(const U32& ref) { // ref excludes the leading '&'
    static const std::unordered_map<std::string, const char*> html5 = [] {
        std::unordered_map<std::string, const char*> m;
        for (const auto& e : unicode_tables::kHtml5)
            m[e.name] = e.utf8;
        return m;
    }();
    if (ref[0] == '#') {
        bool hex = ref.size() > 1 && (ref[1] == 'x' || ref[1] == 'X');
        U32 digits = ref.substr(hex ? 2 : 1);
        if (!digits.empty() && digits.back() == ';')
            digits.pop_back();
        // Python int() has arbitrary precision; anything past 0x10FFFF maps to
        // U+FFFD anyway.
        uint64_t num = 0;
        bool overflow = false;
        for (char32_t d : digits) {
            unsigned v = is_ascii_digit(d) ? d - '0' : (d | 0x20) - 'a' + 10;
            num = num * (hex ? 16 : 10) + v;
            if (num > 0x10FFFFull) {
                overflow = true;
                num = 0x110000;
            }
        }
        for (const auto& e : unicode_tables::kInvalidCharrefs)
            if (!overflow && e.num == num)
                return e.utf8;
        if ((num >= 0xD800 && num <= 0xDFFF) || num > 0x10FFFF)
            return "\xEF\xBF\xBD";
        for (uint32_t c : unicode_tables::kInvalidCodepoints)
            if (c == num)
                return "";
        std::string out;
        utf8_append(out, static_cast<char32_t>(num));
        return out;
    }
    std::string s = utf8_encode(ref);
    if (auto it = html5.find(s); it != html5.end()) {
        return it->second;
    }
    for (size_t x = ref.size() - 1; x > 1; --x) { // longest matching name prefix
        std::string prefix = utf8_encode(ref.substr(0, x));
        if (auto it = html5.find(prefix); it != html5.end()) {
            return std::string(it->second) + utf8_encode(ref.substr(x));
        }
    }
    return "&" + s;
}

U32 html_unescape(const U32& s) {
    // _charref = &(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)
    if (s.find(U'&') == U32::npos)
        return s;
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] != '&') {
            utf8_append(out, s[i++]);
            continue;
        }
        size_t j = i + 1, end = 0;
        if (j < s.size() && s[j] == '#') {
            size_t k = j + 1;
            if (k < s.size() && is_ascii_digit(s[k])) {
                while (k < s.size() && is_ascii_digit(s[k]))
                    ++k;
                end = k;
            } else if (k < s.size() && (s[k] == 'x' || s[k] == 'X') && k + 1 < s.size() && is_ascii_hex(s[k + 1])) {
                k += 1;
                while (k < s.size() && is_ascii_hex(s[k]))
                    ++k;
                end = k;
            }
        } else {
            size_t k = j;
            auto ok = [](char32_t c) {
                return c != '\t' && c != '\n' && c != '\f' && c != ' ' && c != '<' && c != '&' && c != '#' && c != ';';
            };
            while (k < s.size() && k - j < 32 && ok(s[k]))
                ++k;
            if (k > j)
                end = k;
        }
        if (!end) {
            utf8_append(out, s[i++]);
            continue;
        }
        if (end < s.size() && s[end] == ';')
            ++end;
        out += replace_charref(s.substr(j, end - j));
        i = end;
    }
    return utf8_decode(out);
}

// ----------------------------------------------------------------
// pre-tokenization regex
// <\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+
// (IGNORECASE) Applied to already-lowercased text. Under IGNORECASE 's' also
// matches U+017F (long s), which str.lower() keeps unchanged; U+0345 is
// excluded from the "other" class (it case-folds to iota).
bool ci_eq(char32_t c, char32_t lower_ascii) {
    if (c == lower_ascii)
        return true;
    if (lower_ascii == 's' && c == 0x17F)
        return true;
    return false;
}

size_t match_literal(const U32& s, size_t i,
    const char* lit) { // case-insensitive ASCII literal
    size_t n = 0;
    for (; lit[n]; ++n) {
        if (i + n >= s.size())
            return 0;
        char32_t want = static_cast<unsigned char>(lit[n]);
        if (want >= 'a' && want <= 'z') {
            if (!ci_eq(s[i + n], want))
                return 0;
        } else if (s[i + n] != want) {
            return 0;
        }
    }
    return n;
}

bool is_other(char32_t c) { return !is_re_space(c) && !is_letter(c) && !is_number(c) && c != 0x345; }

std::vector<U32> findall(const U32& s) {
    static const char* kLiterals[] = {"<|startoftext|>", "<|endoftext|>", "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    std::vector<U32> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t len = 0;
        for (const char* lit : kLiterals)
            if ((len = match_literal(s, i, lit)))
                break;
        if (!len && is_letter(s[i])) {
            len = 1;
            while (i + len < s.size() && is_letter(s[i + len]))
                ++len;
        }
        if (!len && is_number(s[i]))
            len = 1;
        if (!len && is_other(s[i])) {
            len = 1;
            while (i + len < s.size() && is_other(s[i + len]))
                ++len;
        }
        if (!len) {
            ++i;
            continue;
        } // whitespace (and U+0345): not part of any match
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

// ----------------------------------------------------------------
// bytes_to_unicode
std::vector<char32_t> bytes_to_unicode_order(std::vector<int>& bs) {
    for (int b = '!'; b <= '~'; ++b)
        bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b)
        bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b)
        bs.push_back(b);
    std::vector<char32_t> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; ++b)
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    return cs;
}

std::string pair_key(const std::string& a, const std::string& b) { return a + '\x01' + b; }

// Split a mapped token into its code points (each as a UTF-8 string).
std::vector<std::string> split_chars(const std::string& token) {
    std::vector<std::string> out;
    for (char32_t c : utf8_decode(token)) {
        std::string s;
        utf8_append(s, c);
        out.push_back(std::move(s));
    }
    return out;
}

} // namespace

ClipTokenizer::ClipTokenizer(const std::string& merges_path) {
    std::vector<int> bs;
    std::vector<char32_t> cs = bytes_to_unicode_order(bs);
    byte_encoder_.assign(256, "");
    std::vector<std::string> vocab;
    for (size_t i = 0; i < bs.size(); ++i) {
        std::string u;
        utf8_append(u, cs[i]);
        byte_encoder_[bs[i]] = u;
        vocab.push_back(u);
    }
    for (size_t i = 0, n = vocab.size(); i < n; ++i)
        vocab.push_back(vocab[i] + "</w>");

    std::ifstream f(merges_path);
    if (!f)
        throw std::runtime_error("cannot open " + merges_path);
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos || line.find(' ', sp + 1) != std::string::npos)
            throw std::runtime_error("bad merge line: " + line);
        std::string a = line.substr(0, sp), b = line.substr(sp + 1);
        bpe_ranks_.emplace(pair_key(a, b), rank++);
        vocab.push_back(a + b);
    }
    if (rank != 49152 - 256 - 2)
        throw std::runtime_error("expected 48894 merges, got " + std::to_string(rank));
    vocab.push_back("<|startoftext|>");
    vocab.push_back("<|endoftext|>");
    for (size_t i = 0; i < vocab.size(); ++i)
        encoder_[vocab[i]] = static_cast<int32_t>(i); // later duplicates win, as dict(zip(...))
    cache_["<|startoftext|>"] = "<|startoftext|>";
    cache_["<|endoftext|>"] = "<|endoftext|>";
}

std::string ClipTokenizer::bpe(const std::string& token) const {
    if (auto it = cache_.find(token); it != cache_.end()) {
        return it->second;
    }
    std::vector<std::string> word = split_chars(token);
    word.back() += "</w>";
    if (word.size() == 1)
        return token + "</w>"; // `if not pairs` in Python

    while (true) {
        // bigram = min(pairs, key=rank); unknown pairs have rank inf -> stop
        int best = INT_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = bpe_ranks_.find(pair_key(word[i], word[i + 1]));
            if (it != bpe_ranks_.end() && it->second < best) {
                best = it->second;
                best_i = i;
            }
        }
        if (best == INT_MAX)
            break;
        const std::string first = word[best_i], second = word[best_i + 1];
        std::vector<std::string> merged;
        for (size_t i = 0; i < word.size();) {
            if (word[i] == first && i + 1 < word.size() && word[i + 1] == second) {
                merged.push_back(first + second);
                i += 2;
            } else {
                merged.push_back(word[i++]);
            }
        }
        word.swap(merged);
        if (word.size() == 1)
            break;
    }
    std::string out;
    for (size_t i = 0; i < word.size(); ++i)
        out += (i ? " " : "") + word[i];
    cache_[token] = out;
    return out;
}

std::vector<int32_t> ClipTokenizer::encode(const std::string& text) const {
    // basic_clean: ftfy.fix_text (not ported) -> html.unescape x2 -> strip
    U32 t = py_strip(html_unescape(html_unescape(utf8_decode(text))));
    // whitespace_clean: re.sub(r'\s+', ' ') -> strip ; then .lower()
    U32 ws;
    for (size_t i = 0; i < t.size();) {
        if (is_re_space(t[i])) {
            ws.push_back(U' ');
            while (i < t.size() && is_re_space(t[i]))
                ++i;
        } else {
            ws.push_back(t[i++]);
        }
    }
    t = py_lower(py_strip(ws));

    std::vector<int32_t> ids;
    for (const U32& piece : findall(t)) {
        std::string mapped;
        for (unsigned char b : utf8_encode(piece))
            mapped += byte_encoder_[b];
        std::istringstream ss(bpe(mapped));
        std::string tok;
        while (ss >> tok) {
            auto it = encoder_.find(tok);
            if (it == encoder_.end())
                throw std::runtime_error("token not in vocab: " + tok); // Python raises KeyError
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::vector<int32_t> ClipTokenizer::tokenize(const std::string& text, int context_length) const {
    std::vector<int32_t> tokens{kSot};
    std::vector<int32_t> body = encode(text);
    tokens.insert(tokens.end(), body.begin(), body.end());
    tokens.push_back(kEot);
    if (static_cast<int>(tokens.size()) > context_length) {
        tokens.resize(context_length);
        tokens.back() = kEot;
    }
    tokens.resize(context_length, 0);
    return tokens;
}
