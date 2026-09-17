/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef BENCHMARK_SCENARIO_H
#define BENCHMARK_SCENARIO_H

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "vision_service.h"

namespace vision_benchmark {

struct ScenarioOverrides {
    std::string input_path;
    std::string input_path2;
};

// Supplies task-correct requests to the generic benchmark runner. Expensive
// preparation (image decoding, feature extraction, video decoding and tracker
// initialization) is deliberately kept outside each timed Infer() call.
class BenchmarkScenario {
public:
    virtual ~BenchmarkScenario() = default;

    virtual const char *name() const = 0;
    virtual std::string input_description() const = 0;
    virtual const char *measured_scope() const = 0;

    // Initialize stateful models outside warmup and timed runs.
    virtual bool Initialize(VisionService *service, std::string *error) = 0;

    // Prepare the next request. For video scenarios this decodes the next
    // sequential frame before timing starts.
    virtual bool NextRequest(VisionServiceRequest *request,
                            std::string *error) = 0;

    // Image corresponding to the most recently prepared request. It is used
    // only for separately reported Draw() timing.
    virtual const cv::Mat &draw_image() const = 0;
};

// Reads the optional top-level `benchmark.scenario` declaration and builds a
// scenario. Configs without a declaration use the normal single-image path.
std::unique_ptr<BenchmarkScenario>
CreateBenchmarkScenario(const std::string &config_path,
                        const ScenarioOverrides &overrides,
                        VisionService *service, std::string *error);

} // namespace vision_benchmark

#endif // BENCHMARK_SCENARIO_H
