/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "benchmark_stats.h"

namespace {

bool Near(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

bool Check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
    }
    return condition;
}

}  // namespace

int main() {
    vision_benchmark::ComponentTimingAccumulator stats;
    stats.Add({
        {"detector.infer", 10.0, 1},
        {"recognizer.infer", 6.0, 3},
    });
    stats.Add({
        {"detector.infer", 14.0, 1},
        {"recognizer.infer", 10.0, 5},
    });

    const auto averages = stats.Averages(2);
    if (!Check(averages.size() == 2, "component count changed") ||
        !Check(averages[0].name == "detector.infer", "first-seen order changed") ||
        !Check(Near(averages[0].ms_per_run, 12.0), "detector ms/run is wrong") ||
        !Check(Near(averages[0].calls_per_run, 1.0), "detector calls/run is wrong") ||
        !Check(Near(averages[0].ms_per_call, 12.0), "detector ms/call is wrong") ||
        !Check(averages[1].name == "recognizer.infer", "recognizer order changed") ||
        !Check(Near(averages[1].ms_per_run, 8.0), "recognizer ms/run is wrong") ||
        !Check(Near(averages[1].calls_per_run, 4.0), "recognizer calls/run is wrong") ||
        !Check(Near(averages[1].ms_per_call, 2.0), "recognizer ms/call is wrong")) {
        return 1;
    }

    std::ostringstream component_out;
    vision_benchmark::PrintComponentTimings(component_out, averages);
    const std::string component_text = component_out.str();
    if (!Check(
            component_text.find("Model components:") != std::string::npos,
            "component header missing") ||
        !Check(
            component_text.find("detector.infer:") != std::string::npos,
            "detector component missing") ||
        !Check(
            component_text.find("recognizer.infer:") != std::string::npos,
            "recognizer component missing")) {
        return 1;
    }

    std::ostringstream empty_out;
    vision_benchmark::PrintComponentTimings(empty_out, {});
    if (!Check(empty_out.str().empty(), "empty components produced output") ||
        !Check(stats.Averages(0).empty(), "zero runs should not produce averages")) {
        return 1;
    }

    std::ostringstream tracking_out;
    vision_benchmark::PrintBenchmarkTimingSummary(
        tracking_out, 1.0, 2.0, 3.0, true, 7.0, 1.5);
    const std::string tracking_text = tracking_out.str();
    if (!Check(
            tracking_text.find("Avg model infer: 2") != std::string::npos,
            "tracking omitted model inference") ||
        !Check(
            tracking_text.find("Avg detect: 7") != std::string::npos,
            "tracking omitted detector pipeline") ||
        !Check(
            tracking_text.find("Avg track: 1.5") != std::string::npos,
            "tracking omitted tracker time") ||
        !Check(
            tracking_text.find(
                "Avg detect is the full detector pipeline; "
                "detector.infer is ONNX only") != std::string::npos,
            "tracking semantic note missing")) {
        return 1;
    }
    return 0;
}
