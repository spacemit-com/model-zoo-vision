/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "opencl_handles.h"

#include <utility>

namespace vision_opencl {

OpenClCommandQueue::OpenClCommandQueue(
    cl_command_queue value) noexcept
    : value_(value)
{
}

OpenClCommandQueue::~OpenClCommandQueue()
{
    reset();
}

OpenClCommandQueue::OpenClCommandQueue(
    OpenClCommandQueue&& other) noexcept
    : value_(std::exchange(other.value_, nullptr))
{
}

OpenClCommandQueue&
OpenClCommandQueue::operator=(
    OpenClCommandQueue&& other) noexcept
{
    if (this != &other) {
        reset(std::exchange(other.value_, nullptr));
    }
    return *this;
}

cl_command_queue OpenClCommandQueue::get() const noexcept
{
    return value_;
}

void OpenClCommandQueue::reset(
    cl_command_queue value) noexcept
{
    if (value_) clReleaseCommandQueue(value_);
    value_ = value;
}

OpenClKernel::OpenClKernel(cl_kernel value) noexcept
    : value_(value)
{
}

OpenClKernel::~OpenClKernel()
{
    reset();
}

OpenClKernel::OpenClKernel(
    OpenClKernel&& other) noexcept
    : value_(std::exchange(other.value_, nullptr))
{
}

OpenClKernel& OpenClKernel::operator=(
    OpenClKernel&& other) noexcept
{
    if (this != &other) {
        reset(std::exchange(other.value_, nullptr));
    }
    return *this;
}

cl_kernel OpenClKernel::get() const noexcept
{
    return value_;
}

void OpenClKernel::reset(cl_kernel value) noexcept
{
    if (value_) clReleaseKernel(value_);
    value_ = value;
}

}  // namespace vision_opencl
