/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "external_memory_guard.h"

#include <stdexcept>
#include <utility>

#include "backends/opencl/runtime/opencl_error.h"

namespace vision_opencl {

ExternalMemoryGuard::ExternalMemoryGuard(
    std::shared_ptr<OpenClContext> context,
    cl_command_queue queue,
    std::vector<cl_mem> memories)
    : context_(std::move(context)),
        queue_(queue),
        memories_(std::move(memories))
{
    if (!context_ || queue_ == nullptr || memories_.empty()) {
        throw std::invalid_argument(
            "external memory acquire requires "
            "context, queue, and memory");
    }
    check_cl(
        context_->acquire_external_memory()(
            queue_,
            static_cast<cl_uint>(memories_.size()),
            memories_.data(),
            0,
            nullptr,
            nullptr),
        "acquire external memory");
    acquired_ = true;
}

ExternalMemoryGuard::~ExternalMemoryGuard()
{
    if (!acquired_) return;
    (void)context_->release_external_memory()(
        queue_,
        static_cast<cl_uint>(memories_.size()),
        memories_.data(),
        0,
        nullptr,
        nullptr);
    (void)clFinish(queue_);
}

void ExternalMemoryGuard::release()
{
    if (!acquired_) return;
    check_cl(
        context_->release_external_memory()(
            queue_,
            static_cast<cl_uint>(memories_.size()),
            memories_.data(),
            0,
            nullptr,
            nullptr),
        "release external memory");
    acquired_ = false;
}

}  // namespace vision_opencl
