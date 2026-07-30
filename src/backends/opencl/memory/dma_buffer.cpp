/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dma_buffer.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>

#include <linux/dma-buf.h>
#include <linux/dma-heap.h>

namespace vision_opencl {
namespace {

constexpr size_t kPageSize = 4096;

size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

void dma_sync(int fd, uint64_t flags)
{
    dma_buf_sync sync{flags};
    if (::ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        throw std::runtime_error(
            "DMA_BUF_IOCTL_SYNC failed: " +
            std::string(std::strerror(errno)));
    }
}

}  // namespace

DmaBuffer::DmaBuffer(size_t size)
    : size_(size),
        map_size_(align_up(size, kPageSize))
{
    const char* heaps[] = {
        "/dev/dma_heap/linux,cma",
        "/dev/dma_heap/system"};
    for (const char* path : heaps) {
        const int heap = ::open(path, O_RDWR | O_CLOEXEC);
        if (heap < 0) continue;
        dma_heap_allocation_data allocation{};
        allocation.len = map_size_;
        allocation.fd_flags = O_RDWR | O_CLOEXEC;
        const int result =
            ::ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &allocation);
        ::close(heap);
        if (result == 0) {
            fd_ = allocation.fd;
            break;
        }
    }
    if (fd_ < 0) {
        throw std::runtime_error("failed to allocate output dma-buf");
    }

    data_ = ::mmap(
        nullptr,
        map_size_,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd_,
        0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("failed to mmap output dma-buf");
    }
}

DmaBuffer::~DmaBuffer()
{
    release();
}

DmaBuffer::DmaBuffer(DmaBuffer&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)),
        data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        map_size_(std::exchange(other.map_size_, 0)),
        cpu_read_active_(
            std::exchange(other.cpu_read_active_, false))
{
}

DmaBuffer& DmaBuffer::operator=(DmaBuffer&& other) noexcept
{
    if (this != &other) {
        release();
        fd_ = std::exchange(other.fd_, -1);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        map_size_ = std::exchange(other.map_size_, 0);
        cpu_read_active_ =
            std::exchange(other.cpu_read_active_, false);
    }
    return *this;
}

int DmaBuffer::fd() const noexcept
{
    return fd_;
}

void* DmaBuffer::data() const noexcept
{
    return data_;
}

size_t DmaBuffer::size() const noexcept
{
    return size_;
}

void DmaBuffer::start_cpu_read()
{
    if (cpu_read_active_) return;
    dma_sync(
        fd_,
        DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
    cpu_read_active_ = true;
}

void DmaBuffer::end_cpu_read()
{
    if (!cpu_read_active_) return;
    dma_sync(
        fd_,
        DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
    cpu_read_active_ = false;
}

void DmaBuffer::release() noexcept
{
    if (cpu_read_active_ && fd_ >= 0) {
        dma_buf_sync sync{
            DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ};
        (void)::ioctl(fd_, DMA_BUF_IOCTL_SYNC, &sync);
        cpu_read_active_ = false;
    }
    if (data_ != nullptr) {
        ::munmap(data_, map_size_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    fd_ = -1;
    data_ = nullptr;
    size_ = 0;
    map_size_ = 0;
}

}  // namespace vision_opencl
