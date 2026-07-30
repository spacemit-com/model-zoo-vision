/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENCL_HANDLES_H
#define OPENCL_HANDLES_H

#include <CL/cl.h>

namespace vision_opencl {

class OpenClCommandQueue {
public:
    OpenClCommandQueue() = default;
    explicit OpenClCommandQueue(
        cl_command_queue value) noexcept;
    ~OpenClCommandQueue();

    OpenClCommandQueue(const OpenClCommandQueue&) = delete;
    OpenClCommandQueue& operator=(
        const OpenClCommandQueue&) = delete;
    OpenClCommandQueue(OpenClCommandQueue&& other) noexcept;
    OpenClCommandQueue& operator=(
        OpenClCommandQueue&& other) noexcept;

    cl_command_queue get() const noexcept;
    void reset(
        cl_command_queue value = nullptr) noexcept;

private:
    cl_command_queue value_{nullptr};
};

class OpenClKernel {
public:
    OpenClKernel() = default;
    explicit OpenClKernel(cl_kernel value) noexcept;
    ~OpenClKernel();

    OpenClKernel(const OpenClKernel&) = delete;
    OpenClKernel& operator=(const OpenClKernel&) = delete;
    OpenClKernel(OpenClKernel&& other) noexcept;
    OpenClKernel& operator=(OpenClKernel&& other) noexcept;

    cl_kernel get() const noexcept;
    void reset(cl_kernel value = nullptr) noexcept;

private:
    cl_kernel value_{nullptr};
};

}  // namespace vision_opencl

#endif  // OPENCL_HANDLES_H
