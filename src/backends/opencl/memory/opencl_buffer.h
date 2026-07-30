/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENCL_BUFFER_H
#define OPENCL_BUFFER_H

#include <CL/cl.h>

namespace vision_opencl {

class OpenClBuffer {
public:
    OpenClBuffer() = default;
    explicit OpenClBuffer(cl_mem value) noexcept;
    ~OpenClBuffer();

    OpenClBuffer(const OpenClBuffer&) = delete;
    OpenClBuffer& operator=(const OpenClBuffer&) = delete;
    OpenClBuffer(OpenClBuffer&& other) noexcept;
    OpenClBuffer& operator=(OpenClBuffer&& other) noexcept;

    cl_mem get() const noexcept;
    void reset(cl_mem value = nullptr) noexcept;

private:
    cl_mem value_{nullptr};
};

}  // namespace vision_opencl

#endif  // OPENCL_BUFFER_H
