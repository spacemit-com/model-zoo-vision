/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dma_buffer.h"
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <limits>
#include <stdexcept>

namespace vision_spacemit
{
DmaBuffer::DmaBuffer(size_t bytes)
{
    if (!bytes || bytes > std::numeric_limits<size_t>::max() - 4095)
        throw std::invalid_argument("invalid V2D DMA size");
    size_ = (bytes + 4095) / 4096 * 4096;
    int heap = ::open("/dev/dma_heap/linux,cma", O_RDWR | O_CLOEXEC);
    if (heap < 0) throw std::runtime_error("V2D CMA heap unavailable");
    dma_heap_allocation_data allocation{};
    allocation.len = size_;
    allocation.fd_flags = O_RDWR | O_CLOEXEC;
    int rc = ::ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation);
    ::close(heap);
    if (rc) throw std::runtime_error("V2D DMA allocation failed");
    fd_ = allocation.fd;
    data_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("V2D DMA mapping failed");
    }
}
DmaBuffer::~DmaBuffer()
{
    if (reading_) {
        try {
            end_cpu_read();
        } catch (...) {
        }
    }
    if (data_) ::munmap(data_, size_);
    if (fd_ >= 0) ::close(fd_);
}
void DmaBuffer::begin_cpu_read()
{
    dma_buf_sync s{DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ};
    if (::ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s))
        throw std::runtime_error("V2D output DMA read begin failed");
    reading_ = true;
}
void DmaBuffer::end_cpu_read()
{
    if (!reading_) return;
    dma_buf_sync s{DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ};
    if (::ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s))
        throw std::runtime_error("V2D output DMA read end failed");
    reading_ = false;
}
}  // namespace vision_spacemit
