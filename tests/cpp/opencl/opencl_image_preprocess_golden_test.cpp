/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "common/cpp/image_processing.h"
#include "operators/image_preprocess/image_preprocess_dispatcher.h"
#include "test_dma_buffer.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool write_red_nv12(
    vision_test::TestDmaBuffer& dma,
    int width,
    int height)
{
    if (!dma.begin_write()) return false;
    auto* bytes = static_cast<unsigned char*>(dma.data());
    std::memset(
        bytes, 82,
        static_cast<size_t>(width * height));
    unsigned char* uv = bytes + width * height;
    for (int index = 0;
        index < width * height / 2;
        index += 2) {
        uv[index] = 90;
        uv[index + 1] = 240;
    }
    return dma.end_write();
}

bool write_chroma_edge_nv12(
    vision_test::TestDmaBuffer& dma,
    int width,
    int height)
{
    if (!dma.begin_write()) return false;
    auto* bytes = static_cast<unsigned char*>(dma.data());
    std::memset(
        bytes, 82,
        static_cast<size_t>(width * height));
    unsigned char* uv = bytes + width * height;
    for (int index = 0;
        index < width * height / 2;
        index += 2) {
        const bool red = ((index / 2) & 1) == 0;
        uv[index] = red ? 90 : 240;
        uv[index + 1] = red ? 240 : 110;
    }
    return dma.end_write();
}

bool write_letterbox_pattern_nv12(
    vision_test::TestDmaBuffer& dma,
    int width,
    int height)
{
    if (!dma.begin_write()) return false;
    auto* bytes = static_cast<unsigned char*>(dma.data());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bytes[y * width + x] = static_cast<unsigned char>(
                16 + (x * 3 + y * 5) % 200);
        }
    }
    unsigned char* uv = bytes + width * height;
    for (int y = 0; y < height / 2; ++y) {
        for (int x = 0; x < width; x += 2) {
            uv[y * width + x] =
                static_cast<unsigned char>(
                    96 + ((x / 2) * 7 + y * 3) % 48);
            uv[y * width + x + 1] =
                static_cast<unsigned char>(
                    144 + ((x / 2) * 5 + y * 11) % 64);
        }
    }
    return dma.end_write();
}

void check_color_and_sampling()
{
    using vision_core::ImageInput;
    using vision_core::ImagePixelFormat;
    using vision_operators::ImagePreprocessDispatcher;
    using vision_operators::ImagePreprocessSpec;
    using vision_operators::PreprocessBackendPolicy;

    vision_test::TestDmaBuffer dma(64 * 2 * 3 / 2);
    if (dma.fd() < 0 || !write_red_nv12(dma, 64, 2)) {
        std::cout
            << "SKIP: DMA unavailable for OpenCL color golden\n";
        return;
    }
    ImageInput input;
    input.format = ImagePixelFormat::kNv12;
    input.dma_fd = dma.fd();
    input.image = cv::Mat(3, 64, CV_8UC1, dma.data());

    ImagePreprocessDispatcher dispatcher(
        PreprocessBackendPolicy::kOpenCl);
    ImagePreprocessSpec rgb_spec;
    rgb_spec.output_width = 4;
    rgb_spec.output_height = 4;
    rgb_spec.output_rgb = true;
    auto rgb_result = dispatcher.process(
        input,
        rgb_spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });
    const float* rgb = rgb_result.tensor().ptr<float>();
    constexpr int plane = 16;
    cv::Mat red_bgr;
    cv::cvtColor(
        input.image,
        red_bgr,
        cv::COLOR_YUV2BGR_NV12);
    cv::Mat red_bgr_resized;
    cv::resize(
        red_bgr,
        red_bgr_resized,
        cv::Size(4, 4),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    const cv::Vec3b cpu_red =
        red_bgr_resized.at<cv::Vec3b>(0, 0);
    check(
        std::abs(rgb[0] - cpu_red[2]) <= 2.0F &&
            std::abs(rgb[plane] - cpu_red[1]) <= 2.0F &&
            std::abs(rgb[plane * 2] - cpu_red[0]) <= 2.0F,
        "OpenCL RGB conversion matches OpenCV BT.601");
    rgb_result.complete();

    ImagePreprocessSpec bgr_spec = rgb_spec;
    bgr_spec.output_rgb = false;
    auto bgr_result = dispatcher.process(
        input,
        bgr_spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });
    const float* bgr = bgr_result.tensor().ptr<float>();
    check(
        bgr[0] < 30.0F &&
            bgr[plane] < 30.0F &&
            bgr[plane * 2] > 200.0F,
        "OpenCL BGR output stores red in the third plane");
    bgr_result.complete();

    check(
        write_chroma_edge_nv12(dma, 64, 2),
        "chroma-edge NV12 input is writable");
    ImagePreprocessSpec nearest_spec;
    nearest_spec.output_width = 64;
    nearest_spec.output_height = 2;
    nearest_spec.output_rgb = true;
    nearest_spec.interpolation =
        vision_operators::PreprocessInterpolation::kNearest;
    auto nearest_result = dispatcher.process(
        input,
        nearest_spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });
    const float* nearest = nearest_result.tensor().ptr<float>();
    check(
        nearest[0] > 200.0F &&
            nearest[1] > 200.0F &&
            nearest[2] < 100.0F,
        "nearest sampling reuses one UV texel per 2x2 block");
    nearest_result.complete();

    ImagePreprocessSpec bilinear_spec = nearest_spec;
    bilinear_spec.interpolation =
        vision_operators::PreprocessInterpolation::kBilinear;
    auto bilinear_result = dispatcher.process(
        input,
        bilinear_spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });
    cv::Mat edge_bgr;
    cv::cvtColor(
        input.image,
        edge_bgr,
        cv::COLOR_YUV2BGR_NV12);
    const float* bilinear =
        bilinear_result.tensor().ptr<float>();
    constexpr int edge_plane = 64 * 2;
    bool bilinear_matches_cpu = true;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 64; ++x) {
            const cv::Vec3b cpu =
                edge_bgr.at<cv::Vec3b>(y, x);
            const int index = y * 64 + x;
            bilinear_matches_cpu &=
                std::abs(bilinear[index] - cpu[2]) <= 2.0F &&
                std::abs(
                    bilinear[index + edge_plane] -
                    cpu[1]) <= 2.0F &&
                std::abs(
                    bilinear[index + edge_plane * 2] -
                    cpu[0]) <= 2.0F;
        }
    }
    check(
        bilinear_matches_cpu,
        "bilinear identity preserves OpenCV NV12 chroma blocks");
    bilinear_result.complete();

    ImagePreprocessSpec resized_spec = bilinear_spec;
    resized_spec.output_width = 17;
    resized_spec.output_height = 5;
    auto resized_result = dispatcher.process(
        input,
        resized_spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });
    cv::Mat edge_resized;
    cv::resize(
        edge_bgr,
        edge_resized,
        cv::Size(17, 5),
        0.0,
        0.0,
        cv::INTER_LINEAR);
    const float* resized =
        resized_result.tensor().ptr<float>();
    constexpr int resized_plane = 17 * 5;
    bool resized_matches_cpu = true;
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 17; ++x) {
            const cv::Vec3b cpu =
                edge_resized.at<cv::Vec3b>(y, x);
            const int index = y * 17 + x;
            resized_matches_cpu &=
                std::abs(resized[index] - cpu[2]) <= 3.0F &&
                std::abs(
                    resized[index + resized_plane] -
                    cpu[1]) <= 3.0F &&
                std::abs(
                    resized[index + resized_plane * 2] -
                    cpu[0]) <= 3.0F;
        }
    }
    check(
        resized_matches_cpu,
        "bilinear resize matches OpenCV conversion order");
    resized_result.complete();
}

void check_letterbox()
{
    using vision_core::ImageInput;
    using vision_core::ImagePixelFormat;
    using vision_operators::ImagePreprocessDispatcher;
    using vision_operators::ImagePreprocessSpec;
    using vision_operators::PreprocessBackendPolicy;

    constexpr int width = 64;
    constexpr int height = 32;
    vision_test::TestDmaBuffer dma(width * height * 3 / 2);
    if (dma.fd() < 0 ||
        !write_letterbox_pattern_nv12(dma, width, height)) {
        std::cout
            << "SKIP: DMA unavailable for OpenCL letterbox golden\n";
        return;
    }
    ImageInput input;
    input.format = ImagePixelFormat::kNv12;
    input.dma_fd = dma.fd();
    input.image = cv::Mat(
        height * 3 / 2,
        width,
        CV_8UC1,
        dma.data());

    ImagePreprocessSpec spec;
    spec.output_width = 32;
    spec.output_height = 32;
    spec.resize_mode =
        vision_operators::PreprocessResizeMode::kLetterbox;
    spec.output_rgb = true;
    spec.scale = {
        1.0F / 255.0F,
        1.0F / 255.0F,
        1.0F / 255.0F};
    spec.padding = {114.0F, 114.0F, 114.0F};

    ImagePreprocessDispatcher dispatcher(
        PreprocessBackendPolicy::kOpenCl);
    auto result = dispatcher.process(
        input,
        spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });

    cv::Mat bgr;
    cv::cvtColor(
        input.image,
        bgr,
        cv::COLOR_YUV2BGR_NV12);
    cv::Mat padded = vision_common::letterbox(
        bgr,
        {32, 32},
        cv::Scalar(114, 114, 114));
    cv::Mat cpu_tensor = cv::dnn::blobFromImage(
        padded,
        1.0 / 255.0,
        cv::Size(),
        cv::Scalar(),
        true,
        false,
        CV_32F);

    constexpr int plane = 32 * 32;
    const float expected_padding = 114.0F / 255.0F;
    const float* tensor = result.tensor().ptr<float>();
    bool padding_matches = true;
    for (int channel = 0; channel < 3; ++channel) {
        for (int x = 0; x < 32; ++x) {
            padding_matches &=
                std::abs(
                    tensor[channel * plane + x] -
                    expected_padding) <= 1.0e-6F;
        }
    }
    check(
        padding_matches,
        "OpenCL letterbox writes normalized padding");
    check(
        cv::norm(
            result.tensor(),
            cpu_tensor,
            cv::NORM_INF) <= 4.0 / 255.0,
        "OpenCL letterbox tensor matches CPU golden");
    result.complete();
}

}  // namespace

int main()
{
    check_color_and_sampling();
    check_letterbox();
    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: OpenCL image preprocess golden\n";
    return 0;
}
