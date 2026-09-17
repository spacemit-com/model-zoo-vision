/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "v2d_image_ops.h"
#include "operators/image_preprocess/image_preprocessor.h"
#include "operators/image_preprocess/image_preprocess_geometry.h"
#include "operators/image_preprocess/rvv_image_pack.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sys/stat.h>

namespace vision_operators
{
namespace
{
class V2dImagePreprocessor final : public ImagePreprocessor
{
public:
    explicit V2dImagePreprocessor(const ImagePreprocessSpec& spec) : spec_(spec)
    {
        if (spec.batch_size != 1 ||
            spec.output_type != PreprocessOutputType::kFloat32 ||
            spec.crop_mode != PreprocessCropMode::kNone ||
            spec.interpolation != PreprocessInterpolation::kBilinear)
            throw ImagePreprocessUnsupported(
                "V2D supports batch=1, float32, bilinear, no crop");
        if (spec.output_width <= 0 || spec.output_height <= 0 ||
            spec.output_width > 65535 || spec.output_height > 65535 ||
            size_t(spec.output_width) * spec.output_height >
                std::numeric_limits<int>::max() / 12)
            throw ImagePreprocessUnsupported("V2D output dimensions unsupported");
        for (int c = 0; c < 3; ++c)
            if (!std::isfinite(spec.mean[c]) || !std::isfinite(spec.scale[c]) ||
                !std::isfinite(spec.padding[c]))
                throw std::invalid_argument("nonfinite normalization");
        const int dims[] = {1, 3, spec.output_height, spec.output_width};
        tensor_ = cv::Mat(4, dims, CV_32F);
    }

    cv::Mat process(const vision_core::ImageInput& input) override
    {
        if (active_)
            throw std::invalid_argument(
                "V2D output still in use; complete previous result first");
        if (input.format != vision_core::ImagePixelFormat::kNv12 || input.dma_fd < 0)
            throw ImagePreprocessUnsupported("V2D requires NV12 DMA input");
        int w = input.image.cols, h = input.image.rows * 2 / 3;
        if (input.image.empty() || input.image.type() != CV_8UC1 ||
            input.image.rows % 3 || (w & 1) || (h & 1))
            throw std::invalid_argument("invalid NV12 image view");
        if (w > 65535 || h > 65535 || (input.image.step[0] % 16) != 0)
            throw ImagePreprocessUnsupported(
                "V2D requires 16-byte input stride and 16-bit dimensions");
        if (input.image.isSubmatrix())
            throw ImagePreprocessUnsupported(
                "V2D does not accept offset cv::Mat views");
        const size_t bytes = input.image.step[0] * input.image.rows;
        if (bytes > std::numeric_limits<uint32_t>::max())
            throw ImagePreprocessUnsupported(
                "V2D input allocation exceeds 32-bit size");
        struct stat info{};
        if (::fstat(input.dma_fd, &info) || info.st_size <= 0 ||
            bytes > static_cast<size_t>(info.st_size))
            throw std::invalid_argument("NV12 view exceeds DMA-BUF allocation");
        const auto g = make_image_preprocess_geometry(spec_, w, h);
        if (g.dst_width <= 0 || g.dst_height <= 0 || g.dst_x < 0 || g.dst_y < 0 ||
            g.dst_x + g.dst_width > spec_.output_width ||
            g.dst_y + g.dst_height > spec_.output_height)
            throw std::invalid_argument("V2D output geometry invalid");
        int stride = (g.dst_width * 3 + 15) & ~15;
        size_t size = size_t(stride) * g.dst_height;
        if (!rgb_ || rgb_->size() < size) {
            try {
                rgb_ = std::make_unique<vision_spacemit::DmaBuffer>(size);
            } catch (const std::runtime_error& e) {
                throw ImagePreprocessBackendUnavailable(e.what());
            }
        }
        // All slots are owned by their backend instance; spec changes create a
        // new instance retained by any outstanding result. Geometry changes
        // invalidate padding, including the old content region.
        if (!initialized_ || g.dst_x != geometry_.dst_x || g.dst_y != geometry_.dst_y ||
            g.dst_width != geometry_.dst_width ||
            g.dst_height != geometry_.dst_height) {
            const size_t plane = size_t(spec_.output_width) * spec_.output_height;
            for (int c = 0; c < 3; ++c)
                std::fill(tensor_.ptr<float>() + c * plane,
                            tensor_.ptr<float>() + (c + 1) * plane,
                            (spec_.padding[c] - spec_.mean[c]) * spec_.scale[c]);
            geometry_ = g;
            initialized_ = true;
        }
        try {
            vision_spacemit::resize_nv12_to_rgb(input.image, input.dma_fd, *rgb_,
                                                g.dst_width, g.dst_height, stride);
        } catch (const vision_spacemit::V2dUnavailable& e) {
            throw ImagePreprocessBackendUnavailable(e.what());
        }
        rgb_->begin_cpu_read();
        pack(g, stride);
        rgb_->end_cpu_read();
        active_ = true;
        return tensor_;
    }
    void complete() override { active_ = false; }

private:
    void pack(const ImagePreprocessGeometry& g, int stride)
    {
        const size_t plane = size_t(spec_.output_width) * spec_.output_height;
        for (int y = 0; y < g.dst_height; ++y) {
            const auto* row =
                static_cast<const uint8_t*>(rgb_->data()) + size_t(y) * stride;
            const size_t offset = size_t(g.dst_y + y) * spec_.output_width + g.dst_x;
#if defined(__riscv_vector)
            detail::pack_u8c3_to_f32_planes_rvv(
                row, g.dst_width, !spec_.output_rgb,
                tensor_.ptr<float>() + offset,
                tensor_.ptr<float>() + plane + offset,
                tensor_.ptr<float>() + 2 * plane + offset,
                spec_.mean, spec_.scale);
#else
            for (int x = 0; x < g.dst_width; ++x)
                for (int c = 0; c < 3; ++c) {
                    int source = spec_.output_rgb ? c : 2 - c;
                    tensor_.ptr<float>()[c * plane + offset + x] =
                        (row[x * 3 + source] - spec_.mean[c]) * spec_.scale[c];
                }
#endif
        }
    }
    ImagePreprocessSpec spec_;
    ImagePreprocessGeometry geometry_{};
    cv::Mat tensor_;
    std::unique_ptr<vision_spacemit::DmaBuffer> rgb_;
    bool initialized_ = false;
    bool active_ = false;
};
}  // namespace
std::shared_ptr<ImagePreprocessor> create_v2d_image_preprocessor(
    const ImagePreprocessSpec& spec)
{
    return std::make_shared<V2dImagePreprocessor>(spec);
}
bool v2d_image_preprocessor_compiled() noexcept { return true; }
}  // namespace vision_operators
