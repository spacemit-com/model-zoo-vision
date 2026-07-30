/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EXTERNAL_MEMORY_GUARD_H
#define EXTERNAL_MEMORY_GUARD_H

#include <memory>
#include <vector>

#include "backends/opencl/runtime/opencl_context.h"

namespace vision_opencl {

class ExternalMemoryGuard {
public:
    ExternalMemoryGuard(
        std::shared_ptr<OpenClContext> context,
        cl_command_queue queue,
        std::vector<cl_mem> memories);
    ~ExternalMemoryGuard();

    ExternalMemoryGuard(const ExternalMemoryGuard&) = delete;
    ExternalMemoryGuard& operator=(
        const ExternalMemoryGuard&) = delete;

    void release();

private:
    std::shared_ptr<OpenClContext> context_;
    cl_command_queue queue_{nullptr};
    std::vector<cl_mem> memories_;
    bool acquired_{false};
};

}  // namespace vision_opencl

#endif  // EXTERNAL_MEMORY_GUARD_H
