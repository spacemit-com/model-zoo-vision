/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENCL_CONTEXT_H
#define OPENCL_CONTEXT_H

#include <memory>

#include <CL/cl_ext.h>

#include "opencl_program_cache.h"

namespace vision_opencl {

using ImportMemoryArm = cl_mem(CL_API_CALL*)(
    cl_context,
    cl_mem_flags,
    const cl_import_properties_arm*,
    void*,
    size_t,
    cl_int*);
using AcquireExternalMemory = cl_int(CL_API_CALL*)(
    cl_command_queue,
    cl_uint,
    const cl_mem*,
    cl_uint,
    const cl_event*,
    cl_event*);
using ReleaseExternalMemory = AcquireExternalMemory;

class OpenClContext {
public:
    static std::shared_ptr<OpenClContext> shared();
    ~OpenClContext();

    OpenClContext(const OpenClContext&) = delete;
    OpenClContext& operator=(const OpenClContext&) = delete;

    cl_platform_id platform() const noexcept;
    cl_device_id device() const noexcept;
    cl_context context() const noexcept;
    cl_command_queue create_queue() const;

    ImportMemoryArm import_memory() const noexcept;
    AcquireExternalMemory acquire_external_memory() const noexcept;
    ReleaseExternalMemory release_external_memory() const noexcept;

    OpenClProgramCache& program_cache() noexcept;

private:
    OpenClContext();

    cl_platform_id platform_{nullptr};
    cl_device_id device_{nullptr};
    cl_context context_{nullptr};
    ImportMemoryArm import_memory_{nullptr};
    AcquireExternalMemory acquire_external_memory_{nullptr};
    ReleaseExternalMemory release_external_memory_{nullptr};
    std::unique_ptr<OpenClProgramCache> program_cache_;
};

}  // namespace vision_opencl

#endif  // OPENCL_CONTEXT_H
