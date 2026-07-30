/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cpu_image_preprocessor.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <utility>

#include <linux/dma-buf.h>
#include <opencv2/imgproc.hpp>

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
