/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_stats.h"

#include <utility>

namespace vision_benchmark {

void ComponentTimingAccumulator::Add(
    const std::vector<VisionServiceProfileEntry>& components) {
    for (const auto& component : components) {
        if (component.name.empty()) {
            continue;
        }
        bool found = false;
        for (auto& total : totals_) {
            if (total.name == component.name) {
                total.total_ms += component.total_ms;
                total.calls += component.calls;
                found = true;
                break;
            }
        }
        if (!found) {
            totals_.push_back(component);
        }
    }
}

std::vector<ComponentTimingAverage> ComponentTimingAccumulator::Averages(
    int runs) const {
    if (runs <= 0) {
        return {};
    }

    std::vector<ComponentTimingAverage> averages;
    averages.reserve(totals_.size());
    for (const auto& total : totals_) {
        ComponentTimingAverage average;
        average.name = total.name;
        average.ms_per_run = total.total_ms / static_cast<double>(runs);
        average.calls_per_run =
            static_cast<double>(total.calls) / static_cast<double>(runs);
        average.ms_per_call =
            (total.calls > 0)
                ? total.total_ms / static_cast<double>(total.calls)
                : 0.0;
        averages.push_back(std::move(average));
    }
    return averages;
}

void PrintBenchmarkTimingSummary(
    std::ostream& out,
    double avg_preprocess_ms,
    double avg_model_infer_ms,
    double avg_postprocess_ms,
    bool is_tracking,
    double avg_detect_ms,
    double avg_track_ms) {
    out << "Avg preprocess: " << avg_preprocess_ms << " ms\n"
        << "Avg model infer: " << avg_model_infer_ms << " ms\n"
        << "Avg postprocess: " << avg_postprocess_ms << " ms\n";
    if (is_tracking) {
        out << "Avg detect: " << avg_detect_ms << " ms\n"
            << "Avg track: " << avg_track_ms << " ms\n"
            << "Tracking timing note: Avg detect is the full detector pipeline; "
            << "detector.infer is ONNX only.\n";
    }
}

void PrintComponentTimings(
    std::ostream& out,
    const std::vector<ComponentTimingAverage>& averages) {
    if (averages.empty()) {
        return;
    }
    out << "Model components:\n";
    for (const auto& average : averages) {
        out << "  " << average.name << ": "
            << average.ms_per_run << " ms/run, "
            << average.calls_per_run << " calls/run, "
            << average.ms_per_call << " ms/call\n";
    }
}

}  // namespace vision_benchmark
