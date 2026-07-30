/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "opencl_buffer.h"

#include <utility>

namespace vision_opencl {

OpenClBuffer::OpenClBuffer(cl_mem value) noexcept
    : value_(value)
{
}

OpenClBuffer::~OpenClBuffer()
{
    reset();
}

OpenClBuffer::OpenClBuffer(OpenClBuffer&& other) noexcept
    : value_(std::exchange(other.value_, nullptr))
{
}

OpenClBuffer& OpenClBuffer::operator=(
    OpenClBuffer&& other) noexcept
{
    if (this != &other) {
        reset(std::exchange(other.value_, nullptr));
    }
    return *this;
}

cl_mem OpenClBuffer::get() const noexcept
{
    return value_;
}

void OpenClBuffer::reset(cl_mem value) noexcept
{
    if (value_ != nullptr) {
        clReleaseMemObject(value_);
    }
    value_ = value;
}

}  // namespace vision_opencl
