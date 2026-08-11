/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

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

bool bitwise_equal(const cv::Mat& left, const cv::Mat& right)
{
    return left.type() == right.type() &&
        left.total() == right.total() &&
        left.isContinuous() && right.isContinuous() &&
        std::memcmp(
            left.data,
            right.data,
            left.total() * left.elemSize()) == 0;
}

bool nearly_equal(
    const cv::Mat& left,
    const cv::Mat& right,
    float maximum_difference)
{
    if (left.type() != CV_32F || right.type() != CV_32F ||
        left.total() != right.total() ||
        !left.isContinuous() || !right.isContinuous()) {
        return false;
    }
    for (size_t index = 0; index < left.total(); ++index) {
        if (std::abs(
                left.ptr<float>()[index] - right.ptr<float>()[index]) >
            maximum_difference) {
            return false;
        }
    }
    return true;
}

cv::Mat make_test_image()
{
    cv::Mat image(3, 7, CV_8UC3);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(x * 17 + y * 3),
                static_cast<uint8_t>(x * 7 + y * 29),
                static_cast<uint8_t>(x * 31 + y * 11));
        }
    }
    return image;
}

}  // namespace

int main()
{
    using vision_core::ImageInput;
    using vision_core::ImagePixelFormat;
    using vision_operators::CpuChannelTransform;
    using vision_operators::CpuGrayscaleTransform;
    using vision_operators::PreprocessBackend;
    using vision_operators::ImagePreprocessSpec;
    using vision_operators::PreprocessResizeMode;
    using vision_operators::image_input_to_bgr_cpu;
    using vision_operators::preprocess_bgr_to_nchw;
    using vision_operators::preprocess_bgr_to_gray_nchw;
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

    const cv::Mat test_image = make_test_image();
    ImagePreprocessSpec stretch_spec;
    stretch_spec.output_width = 8;
    stretch_spec.output_height = 6;
    stretch_spec.output_rgb = true;
    stretch_spec.mean = {127.5F, 127.5F, 127.5F};
    stretch_spec.scale = {
        1.0F / 127.5F,
        1.0F / 127.5F,
        1.0F / 127.5F};
    cv::Mat stretch_resized;
    cv::resize(
        test_image,
        stretch_resized,
        cv::Size(8, 6),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    const cv::Mat stretch_reference = cv::dnn::blobFromImage(
        stretch_resized,
        1.0 / 127.5,
        cv::Size(),
        cv::Scalar(127.5, 127.5, 127.5),
        true,
        false,
        CV_32F);
    const cv::Mat stretch_fused =
        preprocess_bgr_to_nchw(test_image, stretch_spec);
    check(
        bitwise_equal(stretch_reference, stretch_fused),
        "fused stretch matches resize plus blobFromImage");

    ImagePreprocessSpec top_left_spec;
    top_left_spec.output_width = 8;
    top_left_spec.output_height = 8;
    top_left_spec.resize_mode = PreprocessResizeMode::kFitTopLeft;
    top_left_spec.output_rgb = true;
    top_left_spec.mean = {
        0.485F * 255.0F,
        0.456F * 255.0F,
        0.406F * 255.0F};
    top_left_spec.scale = {
        1.0F / (0.229F * 255.0F),
        1.0F / (0.224F * 255.0F),
        1.0F / (0.225F * 255.0F)};
    cv::Mat top_left_resized;
    cv::resize(
        test_image,
        top_left_resized,
        cv::Size(8, 3),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    cv::Mat top_left_padded = cv::Mat::zeros(8, 8, CV_8UC3);
    top_left_resized.copyTo(
        top_left_padded(cv::Rect(0, 0, 8, 3)));
    cv::Mat top_left_reference = cv::dnn::blobFromImage(
        top_left_padded,
        1.0 / 255.0,
        cv::Size(),
        cv::Scalar(),
        true,
        false,
        CV_32F);
    const size_t plane_size = 8U * 8U;
    float* reference_values = top_left_reference.ptr<float>();
    const float normalized_mean[] = {0.485F, 0.456F, 0.406F};
    const float standard_deviation[] = {0.229F, 0.224F, 0.225F};
    for (int channel = 0; channel < 3; ++channel) {
        for (size_t index = 0; index < plane_size; ++index) {
            float& value = reference_values[
                static_cast<size_t>(channel) * plane_size + index];
            value = (value - normalized_mean[channel]) /
                standard_deviation[channel];
        }
    }
    CpuChannelTransform top_left_transform;
    top_left_transform.input_scale = {
        1.0F / 255.0F,
        1.0F / 255.0F,
        1.0F / 255.0F};
    top_left_transform.mean = {
        normalized_mean[0], normalized_mean[1], normalized_mean[2]};
    top_left_transform.output_scale = {
        1.0F / standard_deviation[0],
        1.0F / standard_deviation[1],
        1.0F / standard_deviation[2]};
    const cv::Mat top_left_fused =
        preprocess_bgr_to_nchw(
            test_image, top_left_spec, top_left_transform);
    check(
        nearly_equal(top_left_reference, top_left_fused, 1.0e-6F),
        "fused fit-top-left matches padded per-channel normalization");

    ImagePreprocessSpec grayscale_spec;
    grayscale_spec.output_width = 8;
    grayscale_spec.output_height = 6;
    cv::Mat grayscale_resized;
    cv::resize(
        test_image,
        grayscale_resized,
        cv::Size(8, 6),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    cv::Mat float_bgr;
    grayscale_resized.convertTo(float_bgr, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> grayscale_channels;
    cv::split(float_bgr, grayscale_channels);
    cv::Mat grayscale_reference =
        grayscale_channels[2] * 0.299F +
        grayscale_channels[1] * 0.587F +
        grayscale_channels[0] * 0.114F;
    grayscale_reference = grayscale_reference.reshape(1, 1);
    CpuGrayscaleTransform grayscale_transform;
    grayscale_transform.input_scale = 1.0F / 255.0F;
    const cv::Mat grayscale_fused =
        preprocess_bgr_to_gray_nchw(
            test_image, grayscale_spec, grayscale_transform);
    check(
        nearly_equal(grayscale_reference, grayscale_fused, 1.0e-6F),
        "fused grayscale matches resize, conversion, split, and weighting");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: CPU image preprocessor\n";
    return 0;
}
