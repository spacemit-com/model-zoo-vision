/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DMA_BUFFER_H
#define DMA_BUFFER_H
#include <cstddef>

namespace vision_spacemit
{
// CMA-backed storage usable by V2D, independent of OpenCL and model classes.
class DmaBuffer
{
public:
    explicit DmaBuffer(size_t bytes);
    ~DmaBuffer();
    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;
    int fd() const noexcept { return fd_; }
    void* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    void begin_cpu_read();
    void end_cpu_read();

private:
    int fd_ = -1;
    void* data_ = nullptr;
    size_t size_ = 0;
    bool reading_ = false;
};
}  // namespace vision_spacemit

#endif  // DMA_BUFFER_H
