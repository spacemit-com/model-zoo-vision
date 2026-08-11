/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <stdexcept>
#include <string>

#include "operators/image_preprocess/image_preprocess_backend.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main()
{
    using vision_operators::OpenClBackendState;
    using vision_operators::PreprocessBackendPolicy;
    using vision_operators::PreprocessOpenClSampling;
    using vision_operators::parse_preprocess_backend_policy;
    using vision_operators::parse_preprocess_opencl_sampling;

    check(
        parse_preprocess_backend_policy("cpu") ==
            PreprocessBackendPolicy::kCpu,
        "cpu parses as the CPU policy");
    check(
        parse_preprocess_backend_policy("auto") ==
            PreprocessBackendPolicy::kAuto,
        "auto parses as the fallback policy");
    check(
        parse_preprocess_backend_policy("opencl") ==
            PreprocessBackendPolicy::kOpenCl,
        "opencl parses as the strict policy");

    bool rejected_unknown = false;
    try {
        (void)parse_preprocess_backend_policy("vulkan");
    } catch (const std::invalid_argument& error) {
        rejected_unknown =
            std::string(error.what()).find("cpu, auto, or opencl") !=
            std::string::npos;
    }
    check(rejected_unknown, "unknown backend values are rejected");

    check(
        parse_preprocess_opencl_sampling(
            "opencv_compatible") ==
            PreprocessOpenClSampling::kOpenCvCompatible,
        "opencv_compatible parses as compatible sampling");
    check(
        parse_preprocess_opencl_sampling("fast") ==
            PreprocessOpenClSampling::kFast,
        "fast parses as fast sampling");
    bool rejected_unknown_sampling = false;
    try {
        (void)parse_preprocess_opencl_sampling("turbo");
    } catch (const std::invalid_argument& error) {
        rejected_unknown_sampling =
            std::string(error.what()).find(
                "opencv_compatible or fast") !=
            std::string::npos;
    }
    check(
        rejected_unknown_sampling,
        "unknown OpenCL sampling values are rejected");

    OpenClBackendState cpu(PreprocessBackendPolicy::kCpu);
    check(!cpu.should_try_opencl(), "CPU policy never attempts OpenCL");

    OpenClBackendState automatic(PreprocessBackendPolicy::kAuto);
    check(automatic.should_try_opencl(),
            "auto policy initially attempts OpenCL");
    check(
        !automatic.should_try_opencl_for_input(false, false),
        "auto keeps BGR host input on CPU");
    check(
        !automatic.should_try_opencl_for_input(false, true),
        "auto keeps BGR DMA input on CPU");
    check(
        !automatic.should_try_opencl_for_input(true, false),
        "auto keeps NV12 host input on CPU");
    check(
        automatic.should_try_opencl_for_input(true, true),
        "auto attempts OpenCL for NV12 DMA input");
    check(automatic.disable("missing extension"),
            "the first auto failure requests a warning");
    check(!automatic.should_try_opencl(),
            "disabled auto policy no longer attempts OpenCL");
    check(!automatic.disable("second failure"),
            "a disabled backend does not request another warning");
    check(automatic.disable_reason() == "missing extension",
            "the first disable reason remains observable");

    OpenClBackendState strict(PreprocessBackendPolicy::kOpenCl);
    check(strict.should_try_opencl(),
            "strict OpenCL policy attempts OpenCL");
    check(
        !strict.should_try_opencl_for_input(false, false),
        "generic DMA selector excludes BGR host input");
    check(
        !strict.should_try_opencl_for_input(false, true),
        "generic DMA selector excludes BGR DMA input");
    check(
        !strict.should_try_opencl_for_input(true, false),
        "strict OpenCL rejects NV12 host input");
    check(
        strict.should_try_opencl_for_input(true, true),
        "strict OpenCL accepts NV12 DMA input");
    check(!strict.disable("build failed"),
            "strict OpenCL never converts failure into fallback");
    check(strict.should_try_opencl(),
            "strict OpenCL remains strict after a failure");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: image preprocess backend policy\n";
    return 0;
}
