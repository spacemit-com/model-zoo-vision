/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TEST_DMA_BUFFER_H
#define TEST_DMA_BUFFER_H

#include <cstring>
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace vision_test {

class TestDmaBuffer {
public:
    explicit TestDmaBuffer(size_t size) : size_(size)
    {
        const int heap = ::open(
            "/dev/dma_heap/linux,cma",
            O_RDWR | O_CLOEXEC);
        if (heap < 0) return;
        dma_heap_allocation_data allocation{};
        allocation.len = size_;
        allocation.fd_flags = O_RDWR | O_CLOEXEC;
        if (::ioctl(
                heap,
                DMA_HEAP_IOCTL_ALLOC,
                &allocation) == 0) {
            fd_ = allocation.fd;
        }
        ::close(heap);
        if (fd_ < 0) return;
        data_ = ::mmap(
            nullptr,
            size_,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd_,
            0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            return;
        }
        std::memset(data_, 0, size_);
    }

    ~TestDmaBuffer()
    {
        if (data_) ::munmap(data_, size_);
        if (fd_ >= 0) ::close(fd_);
    }

    TestDmaBuffer(const TestDmaBuffer&) = delete;
    TestDmaBuffer& operator=(const TestDmaBuffer&) = delete;

    int fd() const { return fd_; }
    void* data() const { return data_; }

    bool begin_write()
    {
        if (fd_ < 0 || data_ == nullptr) return false;
        dma_buf_sync sync{
            DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE};
        return ::ioctl(fd_, DMA_BUF_IOCTL_SYNC, &sync) == 0;
    }

    bool end_write()
    {
        if (fd_ < 0 || data_ == nullptr) return false;
        dma_buf_sync sync{
            DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE};
        return ::ioctl(fd_, DMA_BUF_IOCTL_SYNC, &sync) == 0;
    }

private:
    size_t size_{0};
    int fd_{-1};
    void* data_{nullptr};
};

}  // namespace vision_test

#endif  // TEST_DMA_BUFFER_H
