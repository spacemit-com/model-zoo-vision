/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VISION_MODEL_CONFIG_H
#define VISION_MODEL_CONFIG_H

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <yaml-cpp/yaml.h>  // NOLINT(build/include_order)

namespace vision_core::yaml_utils {

// Keep supported fields next to each model's parser. Common creation options
// are shared; model-specific and per-call-only options must be explicit.
inline void validateKeys(const YAML::Node& node,
                        std::initializer_list<const char*> keys,
                        const std::string& context,
                        std::initializer_list<const char*> common = {}) {
    if (!node) return;
    if (!node.IsMap()) throw std::invalid_argument(context + " must be a map");
    std::unordered_set<std::string> seen;
    for (const auto& entry : node) {
        if (!entry.first.IsScalar()) {
            throw std::invalid_argument(context + " keys must be strings");
        }
        const std::string key = entry.first.as<std::string>();
        if (!seen.insert(key).second) {
            throw std::invalid_argument("Duplicate configuration field: " + context + "." + key);
        }
        if (std::find(keys.begin(), keys.end(), key) == keys.end() &&
            std::find(common.begin(), common.end(), key) == common.end()) {
            throw std::invalid_argument("Unsupported configuration field: " + context + "." + key);
        }
    }
}

inline void validateModelParams(const YAML::Node& config,
                                std::initializer_list<const char*> keys,
                                std::initializer_list<const char*> preprocess_keys = {}) {
    validateKeys(config["default_params"], keys, "default_params",
        {"num_threads", "providers", "preprocess"});
    const auto params = config["default_params"];
    if (params) {
        validateKeys(params["preprocess"], preprocess_keys,
            "default_params.preprocess", {"backend", "fallback", "opencl_sampling"});
    }
    if (params && params["num_threads"] && params["num_threads"].as<int>() < 0) {
        throw std::invalid_argument("default_params.num_threads must be nonnegative (0 selects ORT defaults)");
    }
}

inline std::string getString(const YAML::Node& node,
                            const std::string& key,
                            const std::string& default_val = "") {
    if (node && node[key]) {
        return node[key].as<std::string>();
    }
    return default_val;
}

inline float getFloat(const YAML::Node& node,
                        const std::string& key,
                        float default_val = 0.0f) {
    if (node && node[key]) {
        const float value = node[key].as<float>();
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Configuration field '" + key + "' must be finite");
        }
        return value;
    }
    return default_val;
}

// Canonical names win when both are supplied. Legacy names remain readable
// while repository YAML files use the same names as inference requests.
inline float getAliasedFloat(const YAML::Node& node,
                            const std::string& key,
                            const std::string& legacy_key,
                            float default_val) {
    return getFloat(node, node && node[key] ? key : legacy_key, default_val);
}

inline int getInt(const YAML::Node& node,
                    const std::string& key,
                    int default_val = 0) {
    if (node && node[key]) {
        return node[key].as<int>();
    }
    return default_val;
}

inline bool getBool(const YAML::Node& node,
                    const std::string& key,
                    bool default_val = false) {
    if (node && node[key]) {
        return node[key].as<bool>();
    }
    return default_val;
}

inline std::string getProvider(const YAML::Node& config,
                                const std::string& default_provider = "SpaceMITExecutionProvider") {
    std::string provider = default_provider;
    if (config && config["default_params"] && config["default_params"]["providers"]) {
        const auto providers = config["default_params"]["providers"];
        if (!providers.IsSequence() || providers.size() != 1 || !providers[0].IsScalar()) {
            throw std::invalid_argument("default_params.providers must contain exactly one provider");
        }
        provider = providers[0].as<std::string>();
    }
    if (provider != "SpaceMITExecutionProvider" && provider != "CPUExecutionProvider") {
        throw std::invalid_argument("Unsupported execution provider: " + provider);
    }
    return provider;
}

}  // namespace vision_core::yaml_utils

#endif  // VISION_MODEL_CONFIG_H
