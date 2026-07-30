/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DMA_BUFFER_H
#define DMA_BUFFER_H

#include <cstddef>

namespace vision_opencl {

class DmaBuffer {
public:
    explicit DmaBuffer(size_t size);
    ~DmaBuffer();

    DmaBuffer(const DmaBuffer&) = delete;
    DmaBuffer& operator=(const DmaBuffer&) = delete;
    DmaBuffer(DmaBuffer&& other) noexcept;
    DmaBuffer& operator=(DmaBuffer&& other) noexcept;

    int fd() const noexcept;
    void* data() const noexcept;
    size_t size() const noexcept;

    void start_cpu_read();
    void end_cpu_read();

private:
    void release() noexcept;

    int fd_{-1};
    void* data_{nullptr};
    size_t size_{0};
    size_t map_size_{0};
    bool cpu_read_active_{false};
};

}  // namespace vision_opencl

#endif  // DMA_BUFFER_H
