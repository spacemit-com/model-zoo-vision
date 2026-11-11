/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vision_session_config.h"

#include <cstddef>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "spacemit_ort_env.h"

namespace vision_core {
namespace {

constexpr char kSpaceMIT[] = "SpaceMITExecutionProvider";
constexpr char kCpu[] = "CPUExecutionProvider";

// Public libspine_tcm.so.3 query ABI, loaded only for EP core discovery.
// Standalone ORT SDKs do not always ship the TCM development header.
struct TcmLayout {
    size_t blk_size;
    size_t blk_num;
    int is_fake_tcm;
};

int available_ai_cores() {
    static const int count = []() -> int {
        const std::unique_ptr<void, int (*)(void*)> library(
            dlopen("libspine_tcm.so.3", RTLD_LAZY | RTLD_LOCAL), &dlclose);
        if (!library) {
            throw std::runtime_error("SpaceMIT EP: libspine_tcm.so.3 is unavailable");
        }
        const auto layout_info = reinterpret_cast<int (*)(TcmLayout*)>(
            dlsym(library.get(), "spine_tcm_runtime_layout_info"));
        TcmLayout layout{};
        if (!layout_info || layout_info(&layout) != 0 ||
            layout.is_fake_tcm || layout.blk_num == 0 ||
            layout.blk_num > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("SpaceMIT EP: no usable TCM core layout");
        }
        return static_cast<int>(layout.blk_num);
    }();
    return count;
}

bool validate_request(const SessionRequest& request) {
    if (request.num_threads < 0) {
        throw std::invalid_argument("Session num_threads must be nonnegative");
    }
    if (request.provider != kSpaceMIT && request.provider != kCpu) {
        throw std::invalid_argument("Unsupported execution provider: " + request.provider);
    }
    return request.provider == kSpaceMIT;
}

}  // namespace

Ort::Env& shared_ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VisionModel");
    return env;
}

std::vector<SessionConfig> plan_session_configs(
    const std::vector<SessionRequest>& requests) {
    bool needs_ep = false;
    for (const auto& request : requests) {
        needs_ep = validate_request(request) || needs_ep;
    }
    return plan_session_configs(requests, needs_ep ? available_ai_cores() : 0);
}

std::vector<SessionConfig> plan_session_configs(
    const std::vector<SessionRequest>& requests,
    int ai_core_count) {
    if (ai_core_count < 0) {
        throw std::invalid_argument("AI core count must be nonnegative");
    }
    std::vector<SessionConfig> result;
    std::vector<size_t> ep_indices;
    for (const auto& request : requests) {
        const bool is_ep = validate_request(request);
        if (is_ep) ep_indices.push_back(result.size());
        result.push_back({request.num_threads, request.provider, is_ep ? 1 : 0});
    }
    if (ep_indices.empty()) return result;
    if (static_cast<size_t>(ai_core_count) < ep_indices.size()) {
        throw std::runtime_error(
            "Not enough AI cores for " + std::to_string(ep_indices.size()) +
            " EP sessions (available: " + std::to_string(ai_core_count) + ")");
    }

    // Max-min fair sizing: each EP session gets at least one thread, and manual
    // limits stop small sessions from taking more than their requested share.
    int remaining = ai_core_count - static_cast<int>(ep_indices.size());
    while (remaining > 0) {
        bool assigned = false;
        for (size_t index = 0; index < ep_indices.size() && remaining > 0; ++index) {
            auto& config = result[ep_indices[index]];
            if (config.num_threads == 0 || config.ep_num_threads < config.num_threads) {
                ++config.ep_num_threads;
                --remaining;
                assigned = true;
            }
        }
        if (!assigned) break;
    }
    return result;
}

void configure_session_options(
    Ort::SessionOptions& options,
    const SessionConfig& config) {
    options.SetIntraOpNumThreads(config.num_threads);
    if (config.provider == kCpu) return;
    if (config.provider != kSpaceMIT || config.ep_num_threads <= 0) {
        throw std::invalid_argument("SpaceMIT EP session has no assigned threads");
    }
    const std::unordered_map<std::string, std::string> provider_options = {
        {"SPACEMIT_EP_INTRA_THREAD_NUM", std::to_string(config.ep_num_threads)},
        {"SPACEMIT_EP_INTER_THREAD_NUM", "1"},
        {"SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD", "0"},
    };
    (void)shared_ort_env();
    Ort::Status status = Ort::SessionOptionsSpaceMITEnvInit(options, provider_options);
    if (!status.IsOK()) {
        throw std::runtime_error(
            "SpaceMIT EP init failed: " + std::string(status.GetErrorMessage()));
    }
    std::cout << "SpaceMIT EP threads: " << config.ep_num_threads << std::endl;
}

}  // namespace vision_core
