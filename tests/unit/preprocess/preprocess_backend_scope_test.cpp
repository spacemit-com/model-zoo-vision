/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/cpp/vision_model_base.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

class TestModel final : public vision_core::BaseModel {
public:
    TestModel() : BaseModel("", true) {}

    void load_model() override {}

    vision_core::InferResponse Run(
        const vision_core::InferRequest&) override
    {
        return {};
    }

    std::vector<vision_core::InferIntent>
    supported_intents() const override
    {
        return {vision_core::InferIntent::kDetect};
    }
};

class EnabledTestModel final : public vision_core::BaseModel {
public:
    EnabledTestModel() : BaseModel("", true)
    {
        enable_accelerated_image_preprocess();
    }

    void load_model() override {}

    vision_core::InferResponse Run(
        const vision_core::InferRequest&) override
    {
        return {};
    }

    std::vector<vision_core::InferIntent>
    supported_intents() const override
    {
        return {vision_core::InferIntent::kDetect};
    }

    PreparedImage prepare_bgr(
        const cv::Mat& image,
        int output_width = 0,
        int output_height = 0)
    {
        vision_operators::ImagePreprocessSpec spec;
        spec.output_width =
            output_width == 0 ? image.cols : output_width;
        spec.output_height =
            output_height == 0 ? image.rows : output_height;
        spec.output_rgb = true;

        vision_core::ImageInput input;
        input.image = image;
        input.format = vision_core::ImagePixelFormat::kBgr8;
        return prepare_image(
            input,
            spec,
            [](const cv::Mat& source) {
                return source.clone();
            });
    }
};

}  // namespace

int main()
{
    TestModel model;
    bool accepted_cpu = true;
    try {
        model.configure_preprocess_backend("cpu");
    } catch (...) {
        accepted_cpu = false;
    }
    check(accepted_cpu, "non-YOLOv8 models accept CPU");

    bool accepted_auto = true;
    try {
        model.configure_preprocess_backend("auto");
    } catch (...) {
        accepted_auto = false;
    }
    check(
        accepted_auto,
        "models without opt-in accept auto as CPU behavior");

    bool rejected = false;
    try {
        model.configure_preprocess_backend("opencl");
    } catch (const std::runtime_error& error) {
        rejected =
            std::string(error.what()).find(
                "does not enable accelerated image preprocessing") !=
            std::string::npos;
    }
    check(
        rejected,
        "models without opt-in reject strict OpenCL");

    EnabledTestModel enabled;
    bool accepted_enabled_auto = true;
    try {
        enabled.configure_preprocess_backend("auto");
    } catch (...) {
        accepted_enabled_auto = false;
    }
    check(
        accepted_enabled_auto,
        "opted-in models accept auto");

    auto prepared = enabled.prepare_bgr(
        cv::Mat::zeros(4, 4, CV_8UC3));
    check(
        prepared.backend_used() ==
            vision_operators::PreprocessBackend::kCpu,
        "auto BGR input uses CPU");
    prepared.complete();

    const vision_core::RuntimeProfile profile =
        enabled.get_runtime_profile();
    bool found_cpu_profile = false;
    for (const auto& component : profile.components) {
        if (component.name == "image_preprocess.cpu" &&
            component.calls == 1) {
            found_cpu_profile = true;
            break;
        }
    }
    check(
        found_cpu_profile,
        "actual CPU preprocess backend is profiled");

    enabled.configure_preprocess_backend("cpu");
    bool cpu_ignored_opencl_only_spec = true;
    try {
        auto dynamic_cpu = enabled.prepare_bgr(
            cv::Mat::zeros(4, 4, CV_8UC3), -1, -1);
        dynamic_cpu.complete();
    } catch (...) {
        cpu_ignored_opencl_only_spec = false;
    }
    check(
        cpu_ignored_opencl_only_spec,
        "CPU policy does not validate OpenCL-only geometry");

    enabled.configure_preprocess_backend("auto");
    bool auto_bgr_ignored_opencl_only_spec = true;
    try {
        auto dynamic_auto = enabled.prepare_bgr(
            cv::Mat::zeros(4, 4, CV_8UC3), -1, -1);
        dynamic_auto.complete();
    } catch (...) {
        auto_bgr_ignored_opencl_only_spec = false;
    }
    check(
        auto_bgr_ignored_opencl_only_spec,
        "auto BGR CPU path does not validate OpenCL-only geometry");

    bool strict_configuration_result = false;
    try {
        enabled.configure_preprocess_backend("opencl");
#if VISION_WITH_OPENCL
        strict_configuration_result = true;
#endif
    } catch (const std::runtime_error& error) {
#if VISION_WITH_OPENCL
        (void)error;
#else
        strict_configuration_result =
            std::string(error.what()).find(
                "was not compiled") != std::string::npos;
#endif
    }
    check(
        strict_configuration_result,
        "strict OpenCL configuration follows build capability");

#if VISION_WITH_OPENCL
    bool accepted_bgr = false;
    try {
        auto prepared_bgr = enabled.prepare_bgr(
            cv::Mat::zeros(4, 4, CV_8UC3));
        accepted_bgr =
            prepared_bgr.backend_used() ==
            vision_operators::PreprocessBackend::kOpenCl;
        prepared_bgr.complete();
    } catch (...) {
        accepted_bgr = false;
    }
    check(
        accepted_bgr,
        "strict OpenCL accepts BGR host input");
#endif

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: preprocess backend scope\n";
    return 0;
}
