/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <opencv2/core.hpp>

#include "operators/image_preprocess/image_preprocess_dispatcher.h"
#include "operators/image_preprocess/image_preprocessor.h"

#if VISION_WITH_OPENCL
#include "test_dma_buffer.h"
#endif

namespace {

int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

#if VISION_WITH_OPENCL
class PreAcquireFailure final
    : public vision_operators::ImagePreprocessor {
public:
    cv::Mat process(
        const vision_core::ImageInput&) override
    {
        ++calls_;
        throw vision_operators::
            ImagePreprocessBackendUnavailable(
                "test import failure");
    }

    void complete() override {}

    int calls() const { return calls_; }

private:
    int calls_{0};
};

class InvalidInputFailure final
    : public vision_operators::ImagePreprocessor {
public:
    cv::Mat process(
        const vision_core::ImageInput&) override
    {
        ++calls_;
        throw std::invalid_argument(
            "test invalid DMA layout");
    }

    void complete() override {}

    int calls() const { return calls_; }

private:
    int calls_{0};
};

class CompletionFailure final
    : public vision_operators::ImagePreprocessor {
public:
    cv::Mat process(
        const vision_core::ImageInput&) override
    {
        const int shape[] = {1, 3, 4, 4};
        return cv::Mat::zeros(4, shape, CV_32F);
    }

    void complete() override
    {
        throw std::runtime_error("test completion failure");
    }
};

class SuccessfulPreprocessor final
    : public vision_operators::ImagePreprocessor {
public:
    cv::Mat process(
        const vision_core::ImageInput&) override
    {
        ++calls_;
        const int shape[] = {1, 3, 4, 4};
        return cv::Mat::zeros(4, shape, CV_32F);
    }

    void complete() override {}

    int calls() const { return calls_; }

private:
    int calls_{0};
};
#endif

}  // namespace

int main()
{
    using vision_core::ImageInput;
    using vision_core::ImagePixelFormat;
    using vision_operators::ImagePreprocessDispatcher;
    using vision_operators::ImagePreprocessSpec;
    using vision_operators::PreprocessBackendPolicy;

    ImageInput input;
    input.format = ImagePixelFormat::kBgr8;
    input.image = cv::Mat::zeros(4, 4, CV_8UC3);

    ImagePreprocessSpec spec;
    spec.output_width = 4;
    spec.output_height = 4;

#if VISION_WITH_OPENCL
    auto successful =
        std::make_shared<SuccessfulPreprocessor>();
    ImagePreprocessDispatcher dispatcher(
        PreprocessBackendPolicy::kOpenCl,
        [successful](const ImagePreprocessSpec&) {
            return successful;
        });
    bool cpu_called = false;
    auto bgr_opencl_result = dispatcher.process(
        input,
        spec,
        [&](const cv::Mat& image) {
            cpu_called = true;
            return image.clone();
        });
    check(
        bgr_opencl_result.backend_used() ==
            vision_operators::PreprocessBackend::kOpenCl &&
            successful->calls() == 1,
        "strict OpenCL accepts BGR host input");
    check(!cpu_called, "strict OpenCL never falls back to CPU");
    bgr_opencl_result.complete();

    auto auto_successful =
        std::make_shared<SuccessfulPreprocessor>();
    ImagePreprocessDispatcher auto_bgr_dispatcher(
        PreprocessBackendPolicy::kAuto,
        [auto_successful](const ImagePreprocessSpec&) {
            return auto_successful;
        });
    bool auto_bgr_cpu_called = false;
    auto auto_bgr_result = auto_bgr_dispatcher.process(
        input,
        spec,
        [&](const cv::Mat& image) {
            auto_bgr_cpu_called = true;
            return image.clone();
        });
    check(
        auto_bgr_cpu_called && auto_successful->calls() == 0 &&
            auto_bgr_result.backend_used() ==
                vision_operators::PreprocessBackend::kCpu,
        "auto keeps BGR host input on CPU");
    auto_bgr_result.complete();

    const int valid_fd = ::open("/dev/null", O_RDONLY);
    if (valid_fd >= 0) {
        ImageInput invalid_layout_input;
        invalid_layout_input.format =
            ImagePixelFormat::kNv12;
        invalid_layout_input.dma_fd = valid_fd;
        invalid_layout_input.image =
            cv::Mat::zeros(6, 4, CV_8UC1);

        auto invalid_input =
            std::make_shared<InvalidInputFailure>();
        ImagePreprocessDispatcher invalid_input_dispatcher(
            PreprocessBackendPolicy::kAuto,
            [invalid_input](const ImagePreprocessSpec&) {
                return invalid_input;
            });
        bool invalid_layout_observed = false;
        bool invalid_layout_cpu_called = false;
        for (int attempt = 0; attempt < 2; ++attempt) {
            try {
                (void)invalid_input_dispatcher.process(
                    invalid_layout_input,
                    spec,
                    [&](const cv::Mat& image) {
                        invalid_layout_cpu_called = true;
                        return image.clone();
                    });
            } catch (const std::invalid_argument& error) {
                invalid_layout_observed =
                    std::string(error.what()).find(
                        "invalid DMA layout") !=
                    std::string::npos;
            }
        }
        check(
            invalid_layout_observed,
            "OpenCL invalid input is reported to the current request");
        check(
            !invalid_layout_cpu_called,
            "OpenCL invalid input never falls back the current request");
        check(
            invalid_input->calls() == 2,
            "OpenCL invalid input preserves later auto attempts");
        ::close(valid_fd);
    } else {
        std::cout
            << "SKIP: /dev/null unavailable for invalid-input test\n";
    }

    vision_test::TestDmaBuffer dma(24);
    if (dma.fd() >= 0) {
        ImageInput nv12_dma;
        nv12_dma.format = ImagePixelFormat::kNv12;
        nv12_dma.dma_fd = dma.fd();
        nv12_dma.image =
            cv::Mat(6, 4, CV_8UC1, dma.data());

        auto pre_acquire_failure =
            std::make_shared<PreAcquireFailure>();
        ImagePreprocessDispatcher fallback_dispatcher(
            PreprocessBackendPolicy::kAuto,
            [pre_acquire_failure](const ImagePreprocessSpec&) {
                return pre_acquire_failure;
            });
        int fallback_cpu_calls = 0;
        std::ostringstream fallback_warning;
        std::streambuf* previous_stderr =
            std::cerr.rdbuf(fallback_warning.rdbuf());
        auto fallback_result = fallback_dispatcher.process(
            nv12_dma,
            spec,
            [&](const cv::Mat& bgr) {
                if (bgr.type() == CV_8UC3) {
                    ++fallback_cpu_calls;
                }
                return bgr.clone();
            });
        std::cerr.rdbuf(previous_stderr);
        check(
            fallback_cpu_calls == 1,
            "pre-acquire OpenCL failure falls back current request");
        check(
            fallback_warning.str().find(
                "subsequent auto requests will use CPU") !=
                std::string::npos,
            "auto-disable warning describes only later requests");
        fallback_result.complete();
        auto later_fallback = fallback_dispatcher.process(
            nv12_dma,
            spec,
            [&](const cv::Mat& bgr) {
                if (bgr.type() == CV_8UC3) {
                    ++fallback_cpu_calls;
                }
                return bgr.clone();
            });
        check(
            fallback_cpu_calls == 2 &&
                pre_acquire_failure->calls() == 1,
            "pre-acquire failure sends later auto requests to CPU");
        later_fallback.complete();

        std::optional<
            vision_operators::ImagePreprocessResult>
            pending_result;
        {
            ImagePreprocessDispatcher lifetime_dispatcher(
                PreprocessBackendPolicy::kAuto,
                [](const ImagePreprocessSpec&) {
                    return std::make_shared<
                        CompletionFailure>();
                });
            pending_result.emplace(
                lifetime_dispatcher.process(
                    nv12_dma,
                    spec,
                    [](const cv::Mat& bgr) {
                        return bgr.clone();
                    }));
        }
        bool completion_error_observed = false;
        try {
            pending_result->complete();
        } catch (const std::runtime_error& error) {
            completion_error_observed =
                std::string(error.what()).find(
                    "completion failure") !=
                std::string::npos;
        }
        check(
            completion_error_observed,
            "preprocess result may outlive dispatcher safely");

        ImagePreprocessSpec batched_spec = spec;
        batched_spec.batch_size = 2;
        ImagePreprocessDispatcher batched_dispatcher(
            PreprocessBackendPolicy::kAuto);
        bool batched_cpu_called = false;
        auto batched_result = batched_dispatcher.process(
            nv12_dma,
            batched_spec,
            [&](const cv::Mat&) {
                batched_cpu_called = true;
                const int shape[] = {2, 3, 4, 4};
                return cv::Mat::zeros(4, shape, CV_32F);
            });
        check(
            batched_cpu_called,
            "auto batched input falls back before OpenCL acquire");
        check(
            batched_result.backend_used() ==
                vision_operators::PreprocessBackend::kCpu,
            "auto batched result reports CPU backend");
        batched_result.complete();
    } else {
        std::cout
            << "SKIP: DMA heap unavailable for dispatcher tests\n";
    }
#else
    ImagePreprocessDispatcher cpu_only_dispatcher(
        PreprocessBackendPolicy::kCpu);
    bool rejected_uncompiled = false;
    try {
        cpu_only_dispatcher.configure("opencl");
    } catch (const std::runtime_error& error) {
        rejected_uncompiled =
            std::string(error.what()).find("not compiled") !=
            std::string::npos;
    }
    check(
        rejected_uncompiled,
        "CPU-only build rejects OpenCL during configuration");
    cpu_only_dispatcher.configure("auto");
    auto cpu_result = cpu_only_dispatcher.process(
        input,
        spec,
        [](const cv::Mat& image) {
            return image.clone();
        });
    check(
        cpu_result.backend_used() ==
            vision_operators::PreprocessBackend::kCpu,
        "CPU-only auto policy stays on CPU");
    cpu_result.complete();
#endif

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: image preprocess dispatcher policy\n";
    return 0;
}
