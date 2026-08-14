/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
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
    using vision_operators::PreprocessOutputType;

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
    const cv::Mat nearest_tensor =
        nearest_result.tensor().clone();
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

    ImagePreprocessSpec identity_letterbox_spec = bilinear_spec;
    identity_letterbox_spec.output_width = 64;
    identity_letterbox_spec.output_height = 4;
    identity_letterbox_spec.resize_mode =
        vision_operators::PreprocessResizeMode::kLetterbox;
    identity_letterbox_spec.padding = {
        114.0F, 114.0F, 114.0F};
    auto identity_letterbox_result = dispatcher.process(
        input,
        identity_letterbox_spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });
    cv::Mat identity_letterbox_cpu = vision_common::letterbox(
        edge_bgr,
        {4, 64},
        cv::Scalar(114, 114, 114));
    cv::Mat identity_letterbox_tensor = cv::dnn::blobFromImage(
        identity_letterbox_cpu,
        1.0,
        cv::Size(),
        cv::Scalar(),
        true,
        false,
        CV_32F);
    check(
        cv::norm(
            identity_letterbox_result.tensor(),
            identity_letterbox_tensor,
            cv::NORM_INF) <= 2.0,
        "packed NV12 identity kernel preserves letterbox padding");
    identity_letterbox_result.complete();

    ImagePreprocessSpec identity_letterbox_fp16_spec =
        identity_letterbox_spec;
    identity_letterbox_fp16_spec.output_type =
        PreprocessOutputType::kFloat16;
    auto identity_letterbox_fp16_result = dispatcher.process(
        input,
        identity_letterbox_fp16_spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });
    cv::Mat identity_letterbox_fp16_as_float;
    identity_letterbox_fp16_result.tensor().convertTo(
        identity_letterbox_fp16_as_float,
        CV_32F);
    check(
        cv::norm(
            identity_letterbox_fp16_as_float,
            identity_letterbox_tensor,
            cv::NORM_INF) <= 2.0,
        "packed NV12 identity kernel supports FP16 output");
    identity_letterbox_fp16_result.complete();

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
    const cv::Mat resized_tensor =
        resized_result.tensor().clone();
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

    ImagePreprocessSpec fast_spec = resized_spec;
    fast_spec.opencl_sampling =
        vision_operators::PreprocessOpenClSampling::kFast;
    auto fast_result = dispatcher.process(
        input,
        fast_spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });
    const float* fast = fast_result.tensor().ptr<float>();
    bool fast_is_finite = true;
    for (int index = 0; index < resized_plane * 3; ++index) {
        fast_is_finite &= std::isfinite(fast[index]);
    }
    const bool fast_smooths_chroma = cv::norm(
        fast_result.tensor(),
        resized_tensor,
        cv::NORM_INF) > 10.0;
    check(
        fast_is_finite,
        "fast OpenCL sampling produces finite output");
    check(
        fast_smooths_chroma,
        "fast OpenCL sampling smooths NV12 chroma before conversion");
    fast_result.complete();

    ImagePreprocessSpec fast_nearest_spec = nearest_spec;
    fast_nearest_spec.opencl_sampling =
        vision_operators::PreprocessOpenClSampling::kFast;
    auto fast_nearest_result = dispatcher.process(
        input,
        fast_nearest_spec,
        [](const cv::Mat& bgr) {
            return bgr.clone();
        });
    check(
        cv::norm(
            fast_nearest_result.tensor(),
            nearest_tensor,
            cv::NORM_INF) == 0.0,
        "fast mode preserves nearest-neighbor sampling");
    fast_nearest_result.complete();
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

void check_bgr_host_input()
{
    using vision_core::ImageInput;
    using vision_core::ImagePixelFormat;
    using vision_operators::ImagePreprocessDispatcher;
    using vision_operators::ImagePreprocessSpec;
    using vision_operators::PreprocessBackend;
    using vision_operators::PreprocessBackendPolicy;
    using vision_operators::PreprocessOutputType;

    // Use an ROI so host row pitch is wider than the active BGR pixels. This
    // verifies that the upload path honors cv::Mat::step instead of assuming
    // every input is tightly packed.
    cv::Mat storage(34, 70, CV_8UC3);
    for (int y = 0; y < storage.rows; ++y) {
        for (int x = 0; x < storage.cols; ++x) {
            storage.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<unsigned char>((x * 3 + y) & 255),
                static_cast<unsigned char>((x + y * 5) & 255),
                static_cast<unsigned char>((x * 7 + y * 2) & 255));
        }
    }
    const cv::Mat bgr = storage(cv::Rect(3, 1, 64, 32));

    ImageInput input;
    input.format = ImagePixelFormat::kBgr8;
    input.image = bgr;

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
    bool cpu_called = false;
    auto result = dispatcher.process(
        input,
        spec,
        [&](const cv::Mat& image) {
            cpu_called = true;
            return image.clone();
        });

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
    check(
        result.backend_used() == PreprocessBackend::kOpenCl &&
            !cpu_called,
        "strict OpenCL executes BGR host input on GPU");
    const double max_difference = cv::norm(
        result.tensor(), cpu_tensor, cv::NORM_INF);
    check(
        max_difference <= 4.0 / 255.0,
        "OpenCL BGR upload and fused preprocess match CPU golden "
        "(max difference=" +
            std::to_string(max_difference) + ")");
    result.complete();

    ImagePreprocessSpec identity_spec = spec;
    identity_spec.output_width = 64;
    identity_spec.output_height = 64;
    auto identity_result = dispatcher.process(
        input,
        identity_spec,
        [](const cv::Mat& image) {
            return image.clone();
        });
    const cv::Mat identity_cpu =
        vision_common::letterbox_to_nchw_rgb_blob(
            bgr,
            {64, 64},
            cv::Scalar(114, 114, 114));
    check(
        cv::norm(
            identity_result.tensor(),
            identity_cpu,
            cv::NORM_INF) <= 1.0e-6,
        "OpenCL BGR identity sampling matches CPU no-resize path");
    identity_result.complete();

    ImagePreprocessSpec identity_bgr_spec = identity_spec;
    identity_bgr_spec.output_rgb = false;
    auto identity_bgr_result = dispatcher.process(
        input,
        identity_bgr_spec,
        [](const cv::Mat& image) {
            return image.clone();
        });
    cv::Mat identity_bgr_cpu = identity_cpu.clone();
    constexpr int identity_plane = 64 * 64;
    float* identity_bgr_data = identity_bgr_cpu.ptr<float>();
    std::swap_ranges(
        identity_bgr_data,
        identity_bgr_data + identity_plane,
        identity_bgr_data + identity_plane * 2);
    const double identity_bgr_max_difference = cv::norm(
        identity_bgr_result.tensor(),
        identity_bgr_cpu,
        cv::NORM_INF);
    check(
        identity_bgr_max_difference <= 1.0e-6,
        "OpenCL packed BGR identity kernel supports BGR NCHW output "
        "(max difference=" +
            std::to_string(identity_bgr_max_difference) + ")");
    identity_bgr_result.complete();

    ImagePreprocessSpec identity_fp16_spec = identity_spec;
    identity_fp16_spec.output_type = PreprocessOutputType::kFloat16;
    auto identity_fp16_result = dispatcher.process(
        input,
        identity_fp16_spec,
        [](const cv::Mat& image) {
            return image.clone();
        });
    cv::Mat identity_fp16_as_float;
    identity_fp16_result.tensor().convertTo(
        identity_fp16_as_float, CV_32F);
    check(
        cv::norm(
            identity_fp16_as_float,
            identity_cpu,
            cv::NORM_INF) <= 1.0e-3,
        "OpenCL packed BGR identity kernel supports FP16 output");
    identity_fp16_result.complete();

    // Exercise packed four-pixel groups with one padding pixel on each side.
    // The two boundary groups take the scalar tail path while all interior
    // groups use the 12-byte packed BGR load.
    const cv::Mat offset_bgr = bgr(cv::Rect(0, 0, 62, 32));
    ImageInput offset_input;
    offset_input.format = ImagePixelFormat::kBgr8;
    offset_input.image = offset_bgr;
    ImagePreprocessSpec offset_spec = spec;
    offset_spec.output_width = 64;
    offset_spec.output_height = 32;
    auto offset_result = dispatcher.process(
        offset_input,
        offset_spec,
        [](const cv::Mat& image) {
            return image.clone();
        });
    const cv::Mat offset_cpu =
        vision_common::letterbox_to_nchw_rgb_blob(
            offset_bgr,
            {32, 64},
            cv::Scalar(114, 114, 114));
    const double offset_max_difference = cv::norm(
        offset_result.tensor(), offset_cpu, cv::NORM_INF);
    check(
        offset_max_difference <= 1.0e-6,
        "OpenCL packed BGR identity kernel handles offset boundaries "
        "(max difference=" +
            std::to_string(offset_max_difference) + ")");
    offset_result.complete();

    // Odd output dimensions make both row and plane offsets unaligned. The
    // optimized kernel must use its scalar fallback instead of relying on
    // driver behavior for unaligned vector stores.
    const cv::Mat odd_bgr = bgr(cv::Rect(0, 0, 63, 32));
    ImageInput odd_input;
    odd_input.format = ImagePixelFormat::kBgr8;
    odd_input.image = odd_bgr;
    ImagePreprocessSpec odd_spec = spec;
    odd_spec.output_width = 63;
    odd_spec.output_height = 33;
    auto odd_result = dispatcher.process(
        odd_input,
        odd_spec,
        [](const cv::Mat& image) {
            return image.clone();
        });
    const cv::Mat odd_cpu =
        vision_common::letterbox_to_nchw_rgb_blob(
            odd_bgr,
            {33, 63},
            cv::Scalar(114, 114, 114));
    check(
        cv::norm(
            odd_result.tensor(),
            odd_cpu,
            cv::NORM_INF) <= 1.0e-6,
        "OpenCL packed BGR identity kernel handles unaligned output");
    odd_result.complete();
}

void check_area_rejected()
{
    vision_operators::ImagePreprocessSpec spec;
    spec.output_width = 32;
    spec.output_height = 32;
    spec.interpolation =
        vision_operators::PreprocessInterpolation::kArea;
    bool rejected = false;
    try {
        (void)vision_operators::create_opencl_image_preprocessor(spec);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find(
            "does not support area interpolation") != std::string::npos;
    }
    check(
        rejected,
        "OpenCL explicitly rejects unsupported area interpolation");
}

}  // namespace

int main()
{
    check_color_and_sampling();
    check_letterbox();
    check_bgr_host_input();
    check_area_rejected();
    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: OpenCL image preprocess golden\n";
    return 0;
}
