/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cpu_image_preprocessor.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <utility>

#include <linux/dma-buf.h>
#include <opencv2/imgproc.hpp>

#include "image_preprocess_geometry.h"

namespace vision_operators {
namespace {

void dma_sync(int fd, uint64_t flags)
{
    dma_buf_sync sync{flags};
    if (::ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        throw std::runtime_error(
            "DMA_BUF_IOCTL_SYNC failed: " +
            std::string(std::strerror(errno)));
    }
}

class DmaCpuReadGuard {
public:
    explicit DmaCpuReadGuard(int fd) : fd_(fd)
    {
        if (fd_ < 0) return;
        dma_sync(
            fd_,
            DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
        active_ = true;
    }

    ~DmaCpuReadGuard()
    {
        if (!active_) return;
        dma_buf_sync sync{
            DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ};
        (void)::ioctl(fd_, DMA_BUF_IOCTL_SYNC, &sync);
    }

    DmaCpuReadGuard(const DmaCpuReadGuard&) = delete;
    DmaCpuReadGuard& operator=(
        const DmaCpuReadGuard&) = delete;

    void complete()
    {
        if (!active_) return;
        dma_sync(
            fd_,
            DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
        active_ = false;
    }

private:
    int fd_{-1};
    bool active_{false};
};

struct CpuPreprocessScratch {
    cv::Mat intermediate;
    cv::Mat resized;
};

cv::Mat crop_source(
    const cv::Mat& bgr,
    const ImagePreprocessSpec& spec,
    CpuPreprocessScratch* scratch)
{
    if (spec.crop_mode == PreprocessCropMode::kNone) {
        return bgr;
    }
    if (spec.crop_mode == PreprocessCropMode::kCenterSquare) {
        const int side = std::min(bgr.cols, bgr.rows);
        const int x = (bgr.cols - side) / 2;
        const int y = (bgr.rows - side) / 2;
        return bgr(cv::Rect(x, y, side, side));
    }
    if (spec.resize_width <= 0 || spec.resize_height <= 0 ||
        spec.output_width > spec.resize_width ||
        spec.output_height > spec.resize_height) {
        throw std::invalid_argument(
            "center-crop preprocessing dimensions are invalid");
    }
    const int interpolation =
        spec.interpolation == PreprocessInterpolation::kNearest
        ? cv::INTER_NEAREST
        : cv::INTER_LINEAR;
    cv::resize(
        bgr,
        scratch->intermediate,
        cv::Size(spec.resize_width, spec.resize_height),
        0.0,
        0.0,
        interpolation);
    const int x = (spec.resize_width - spec.output_width) / 2;
    const int y = (spec.resize_height - spec.output_height) / 2;
    return scratch->intermediate(
        cv::Rect(x, y, spec.output_width, spec.output_height));
}

cv::Mat resize_source(
    const cv::Mat& bgr,
    const ImagePreprocessSpec& spec,
    const ImagePreprocessGeometry& geometry,
    CpuPreprocessScratch* scratch)
{
    const cv::Mat source = crop_source(bgr, spec, scratch);
    if (source.cols == geometry.dst_width &&
        source.rows == geometry.dst_height) {
        return source;
    }
    const int interpolation =
        spec.interpolation == PreprocessInterpolation::kNearest
        ? cv::INTER_NEAREST
        : cv::INTER_LINEAR;
    cv::resize(
        source,
        scratch->resized,
        cv::Size(geometry.dst_width, geometry.dst_height),
        0.0,
        0.0,
        interpolation);
    return scratch->resized;
}

void fill_padding(
    float* plane,
    int output_width,
    int output_height,
    const ImagePreprocessGeometry& geometry,
    float value)
{
    const int content_right = geometry.dst_x + geometry.dst_width;
    const int content_bottom = geometry.dst_y + geometry.dst_height;
    if (geometry.dst_y > 0) {
        std::fill(
            plane,
            plane + static_cast<size_t>(geometry.dst_y) * output_width,
            value);
    }
    if (content_bottom < output_height) {
        std::fill(
            plane + static_cast<size_t>(content_bottom) * output_width,
            plane + static_cast<size_t>(output_height) * output_width,
            value);
    }
    if (geometry.dst_x > 0 || content_right < output_width) {
        for (int y = geometry.dst_y; y < content_bottom; ++y) {
            float* row = plane + static_cast<size_t>(y) * output_width;
            std::fill(row, row + geometry.dst_x, value);
            std::fill(row + content_right, row + output_width, value);
        }
    }
}

}  // namespace

cv::Mat image_input_to_bgr_cpu(
    const vision_core::ImageInput& input)
{
    if (input.format == vision_core::ImagePixelFormat::kBgr8) {
        if (input.dma_fd < 0) {
            return input.image;
        }
        DmaCpuReadGuard read_guard(input.dma_fd);
        cv::Mat owned = input.image.clone();
        read_guard.complete();
        return owned;
    }
    DmaCpuReadGuard read_guard(input.dma_fd);
    cv::Mat bgr;
    cv::cvtColor(input.image, bgr, cv::COLOR_YUV2BGR_NV12);
    read_guard.complete();
    return bgr;
}

cv::Mat preprocess_bgr_to_nchw(
    const cv::Mat& bgr,
    const ImagePreprocessSpec& spec,
    const CpuChannelTransform& transform)
{
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument(
            "CPU fused preprocessing requires a non-empty BGR8 image");
    }
    if (spec.batch_size != 1 ||
        spec.output_width <= 0 || spec.output_height <= 0 ||
        spec.output_type != PreprocessOutputType::kFloat32) {
        throw std::invalid_argument(
            "CPU fused preprocessing requires float32 [1,3,H,W] output");
    }
    const ImagePreprocessGeometry geometry =
        make_image_preprocess_geometry(
            spec, bgr.cols, bgr.rows);
    if (geometry.dst_x < 0 || geometry.dst_y < 0 ||
        geometry.dst_width <= 0 || geometry.dst_height <= 0 ||
        geometry.dst_x + geometry.dst_width > spec.output_width ||
        geometry.dst_y + geometry.dst_height > spec.output_height) {
        throw std::invalid_argument(
            "CPU fused preprocessing produced invalid geometry");
    }

    thread_local CpuPreprocessScratch scratch;
    const cv::Mat resized =
        resize_source(bgr, spec, geometry, &scratch);

    const int dimensions[] = {
        1, 3, spec.output_height, spec.output_width};
    cv::Mat tensor(4, dimensions, CV_32F);
    float* output = tensor.ptr<float>();
    const size_t plane_size =
        static_cast<size_t>(spec.output_width) * spec.output_height;
    std::array<std::array<float, 256>, 3> channel_lut{};
    for (int channel = 0; channel < 3; ++channel) {
        for (int value = 0; value < 256; ++value) {
            // Preserve the reference pipeline's rounding point between its
            // uint8 scaling and normalization stages. The volatile temporary
            // prevents contraction into a fused multiply-add; this work is
            // performed only for the 768 LUT entries, not for every pixel.
            volatile float scaled_input =
                value * transform.input_scale[channel];
            channel_lut[channel][value] =
                (scaled_input - transform.mean[channel]) *
                transform.output_scale[channel];
        }
        volatile float scaled_padding =
            spec.padding[channel] * transform.input_scale[channel];
        fill_padding(
            output + static_cast<size_t>(channel) * plane_size,
            spec.output_width,
            spec.output_height,
            geometry,
            (scaled_padding - transform.mean[channel]) *
                transform.output_scale[channel]);
    }

    const auto pack_rows = [&](const cv::Range& rows) {
        float* first_plane = output;
        float* second_plane = output + plane_size;
        float* third_plane = output + plane_size * 2U;
        const int first_source = spec.output_rgb ? 2 : 0;
        const int third_source = spec.output_rgb ? 0 : 2;
        for (int y = rows.start; y < rows.end; ++y) {
            const uint8_t* source_row = resized.ptr<uint8_t>(y);
            const size_t offset =
                static_cast<size_t>(geometry.dst_y + y) *
                    spec.output_width +
                geometry.dst_x;
            float* first = first_plane + offset;
            float* second = second_plane + offset;
            float* third = third_plane + offset;
            for (int x = 0; x < geometry.dst_width; ++x) {
                const uint8_t* pixel = source_row + x * 3;
                first[x] = channel_lut[0][pixel[first_source]];
                second[x] = channel_lut[1][pixel[1]];
                third[x] = channel_lut[2][pixel[third_source]];
            }
        }
    };
    constexpr int kParallelPixelThreshold = 64 * 1024;
    if (geometry.dst_width * geometry.dst_height >=
        kParallelPixelThreshold) {
        cv::parallel_for_(
            cv::Range(0, geometry.dst_height), pack_rows);
    } else {
        pack_rows(cv::Range(0, geometry.dst_height));
    }
    return tensor;
}

cv::Mat preprocess_bgr_to_nchw(
    const cv::Mat& bgr,
    const ImagePreprocessSpec& spec)
{
    CpuChannelTransform transform;
    transform.mean = spec.mean;
    transform.output_scale = spec.scale;
    return preprocess_bgr_to_nchw(bgr, spec, transform);
}

cv::Mat preprocess_bgr_to_gray_nchw(
    const cv::Mat& bgr,
    const ImagePreprocessSpec& spec,
    const CpuGrayscaleTransform& transform)
{
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument(
            "CPU fused grayscale preprocessing requires a non-empty BGR8 "
            "image");
    }
    if (spec.batch_size != 1 ||
        spec.output_width <= 0 || spec.output_height <= 0 ||
        spec.output_type != PreprocessOutputType::kFloat32) {
        throw std::invalid_argument(
            "CPU fused grayscale preprocessing requires float32 "
            "[1,1,H,W] output");
    }
    const ImagePreprocessGeometry geometry =
        make_image_preprocess_geometry(spec, bgr.cols, bgr.rows);
    if (geometry.dst_x < 0 || geometry.dst_y < 0 ||
        geometry.dst_width <= 0 || geometry.dst_height <= 0 ||
        geometry.dst_x + geometry.dst_width > spec.output_width ||
        geometry.dst_y + geometry.dst_height > spec.output_height) {
        throw std::invalid_argument(
            "CPU fused grayscale preprocessing produced invalid geometry");
    }

    thread_local CpuPreprocessScratch scratch;
    const cv::Mat resized =
        resize_source(bgr, spec, geometry, &scratch);
    const int dimensions[] = {
        1, 1, spec.output_height, spec.output_width};
    cv::Mat tensor(4, dimensions, CV_32F);
    float* output = tensor.ptr<float>();

    std::array<float, 3> weighted_padding{};
    for (int channel = 0; channel < 3; ++channel) {
        volatile float scaled_padding =
            spec.padding[channel] * transform.input_scale;
        weighted_padding[channel] =
            scaled_padding * transform.bgr_weights[channel];
    }
    volatile float padding_red_green =
        weighted_padding[2] + weighted_padding[1];
    const float padding =
        ((padding_red_green + weighted_padding[0]) - transform.mean) *
        transform.output_scale;
    fill_padding(
        output,
        spec.output_width,
        spec.output_height,
        geometry,
        padding);

    std::array<std::array<float, 256>, 3> channel_lut{};
    for (int channel = 0; channel < 3; ++channel) {
        for (int value = 0; value < 256; ++value) {
            volatile float scaled_input =
                value * transform.input_scale;
            channel_lut[channel][value] =
                scaled_input * transform.bgr_weights[channel];
        }
    }

    const auto pack_rows = [&](const cv::Range& rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            const uint8_t* source_row = resized.ptr<uint8_t>(y);
            float* destination =
                output +
                static_cast<size_t>(geometry.dst_y + y) *
                    spec.output_width +
                geometry.dst_x;
            for (int x = 0; x < geometry.dst_width; ++x) {
                const uint8_t* pixel = source_row + x * 3;
                volatile float red_green =
                    channel_lut[2][pixel[2]] +
                    channel_lut[1][pixel[1]];
                destination[x] =
                    ((red_green + channel_lut[0][pixel[0]]) -
                        transform.mean) *
                    transform.output_scale;
            }
        }
    };
    constexpr int kParallelPixelThreshold = 64 * 1024;
    if (geometry.dst_width * geometry.dst_height >=
        kParallelPixelThreshold) {
        cv::parallel_for_(
            cv::Range(0, geometry.dst_height), pack_rows);
    } else {
        pack_rows(cv::Range(0, geometry.dst_height));
    }
    return tensor;
}

ImagePreprocessResult run_cpu_image_preprocess(
    const vision_core::ImageInput& input,
    const CpuImagePreprocess& cpu_preprocess)
{
    if (!cpu_preprocess) {
        throw std::invalid_argument(
            "CPU image preprocess callback is empty");
    }
    if (input.format == vision_core::ImagePixelFormat::kBgr8) {
        DmaCpuReadGuard read_guard(input.dma_fd);
        cv::Mat tensor = cpu_preprocess(input.image);
        read_guard.complete();
        return ImagePreprocessResult(
            std::move(tensor),
            PreprocessBackend::kCpu);
    }
    return ImagePreprocessResult(
        cpu_preprocess(image_input_to_bgr_cpu(input)),
        PreprocessBackend::kCpu);
}

}  // namespace vision_operators
