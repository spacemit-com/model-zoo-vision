/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef BENCHMARK_STATS_H
#define BENCHMARK_STATS_H

#include <ostream>
#include <string>
#include <vector>

#include "vision_service.h"

namespace vision_benchmark {

struct ComponentTimingAverage {
    std::string name;
    double ms_per_run = 0.0;
    double calls_per_run = 0.0;
    double ms_per_call = 0.0;
};

class ComponentTimingAccumulator {
public:
    void Add(const std::vector<VisionServiceProfileEntry>& components);
    std::vector<ComponentTimingAverage> Averages(int runs) const;

private:
    std::vector<VisionServiceProfileEntry> totals_;
};

void PrintBenchmarkTimingSummary(
    std::ostream& out,
    double avg_preprocess_ms,
    double avg_model_infer_ms,
    double avg_postprocess_ms,
    bool is_tracking,
    double avg_detect_ms,
    double avg_track_ms);

void PrintComponentTimings(
    std::ostream& out,
    const std::vector<ComponentTimingAverage>& averages);

}  // namespace vision_benchmark

#endif  // BENCHMARK_STATS_H
