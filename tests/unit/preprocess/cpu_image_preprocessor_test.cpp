/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include <opencv2/core.hpp>

#include "operators/image_preprocess/cpu_image_preprocessor.h"

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
    using vision_core::ImageInput;
    using vision_core::ImagePixelFormat;
    using vision_operators::PreprocessBackend;
    using vision_operators::image_input_to_bgr_cpu;
    using vision_operators::run_cpu_image_preprocess;

    ImageInput bgr;
    bgr.format = ImagePixelFormat::kBgr8;
    bgr.image = cv::Mat::zeros(4, 4, CV_8UC3);
    bool saw_same_bgr = false;
    auto bgr_result = run_cpu_image_preprocess(
        bgr,
        [&](const cv::Mat& image) {
            saw_same_bgr = image.data == bgr.image.data;
            return image.clone();
        });
    check(saw_same_bgr, "BGR CPU input is not converted");
    check(
        bgr_result.backend_used() == PreprocessBackend::kCpu,
        "BGR result reports CPU backend");

    ImageInput bgr_dma = bgr;
    bgr_dma.dma_fd = ::open("/dev/null", O_RDONLY);
    bool rejected_unsynchronizable_dma = false;
    bool dma_callback_called = false;
    try {
        (void)run_cpu_image_preprocess(
            bgr_dma,
            [&](const cv::Mat& image) {
                dma_callback_called = true;
                return image.clone();
            });
    } catch (const std::runtime_error& error) {
        rejected_unsynchronizable_dma =
            std::string(error.what()).find(
                "DMA_BUF_IOCTL_SYNC") != std::string::npos;
    }
    ::close(bgr_dma.dma_fd);
    check(
        rejected_unsynchronizable_dma,
        "BGR DMA input enters CPU cache synchronization");
    check(
        !dma_callback_called,
        "BGR DMA callback waits for cache synchronization");

    ImageInput nv12;
    nv12.format = ImagePixelFormat::kNv12;
    nv12.image = cv::Mat::zeros(6, 4, CV_8UC1);
    bool saw_converted_bgr = false;
    auto nv12_result = run_cpu_image_preprocess(
        nv12,
        [&](const cv::Mat& image) {
            saw_converted_bgr =
                image.type() == CV_8UC3 &&
                image.rows == 4 &&
                image.cols == 4;
            return image.clone();
        });
    check(
        saw_converted_bgr,
        "NV12 CPU fallback adapts input to BGR");
    check(
        nv12_result.backend_used() == PreprocessBackend::kCpu,
        "NV12 fallback reports CPU backend");

    const cv::Mat converted = image_input_to_bgr_cpu(nv12);
    check(
        converted.type() == CV_8UC3 &&
            converted.rows == 4 &&
            converted.cols == 4,
        "NV12 input can be adapted for CPU postprocessing");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: CPU image preprocessor\n";
    return 0;
}
