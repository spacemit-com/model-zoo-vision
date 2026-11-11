/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VISION_SESSION_CONFIG_H
#define VISION_SESSION_CONFIG_H

#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>  // NOLINT(build/include_order)

namespace vision_core {

struct SessionRequest {
    int num_threads;
    std::string provider;
};

struct SessionConfig {
    // ORT CPU operator threads retain the original request, including zero.
    int num_threads;
    std::string provider;
    int ep_num_threads;
};

Ort::Env& shared_ort_env();

// One model's EP sessions share a thread budget based on the TCM core count.
// CPU sessions keep their requested threads and consume no AI core budget.
// Zero requests automatic sizing. Physical CPU affinity is left to the EP.
std::vector<SessionConfig> plan_session_configs(
    const std::vector<SessionRequest>& requests);

// Explicit budget overload keeps the allocation policy independent of TCM.
std::vector<SessionConfig> plan_session_configs(
    const std::vector<SessionRequest>& requests,
    int ai_core_count);

void configure_session_options(
    Ort::SessionOptions& options,
    const SessionConfig& config);

}  // namespace vision_core

#endif  // VISION_SESSION_CONFIG_H
