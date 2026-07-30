/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "opencl_context.h"

#include <stdexcept>

#include "opencl_error.h"

namespace vision_opencl {

std::shared_ptr<OpenClContext> OpenClContext::shared()
{
    static std::shared_ptr<OpenClContext> instance(
        new OpenClContext());
    return instance;
}

OpenClContext::OpenClContext()
{
    cl_int error = clGetPlatformIDs(1, &platform_, nullptr);
    check_cl(error, "clGetPlatformIDs");
    error = clGetDeviceIDs(
        platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
    check_cl(error, "clGetDeviceIDs");
    context_ = clCreateContext(
        nullptr, 1, &device_, nullptr, nullptr, &error);
    check_cl(error, "clCreateContext");

    import_memory_ = reinterpret_cast<ImportMemoryArm>(
        clGetExtensionFunctionAddressForPlatform(
            platform_, "clImportMemoryARM"));
    acquire_external_memory_ =
        reinterpret_cast<AcquireExternalMemory>(
            clGetExtensionFunctionAddressForPlatform(
                platform_,
                "clEnqueueAcquireExternalMemObjectsKHR"));
    release_external_memory_ =
        reinterpret_cast<ReleaseExternalMemory>(
            clGetExtensionFunctionAddressForPlatform(
                platform_,
                "clEnqueueReleaseExternalMemObjectsKHR"));
    if (!import_memory_ ||
        !acquire_external_memory_ ||
        !release_external_memory_) {
        clReleaseContext(context_);
        context_ = nullptr;
        throw std::runtime_error(
            "OpenCL DMA-BUF import/acquire/release is unavailable");
    }

    program_cache_ = std::make_unique<OpenClProgramCache>(
        context_, device_);
}

OpenClContext::~OpenClContext()
{
    program_cache_.reset();
    if (context_ != nullptr) {
        clReleaseContext(context_);
    }
}

cl_platform_id OpenClContext::platform() const noexcept
{
    return platform_;
}

cl_device_id OpenClContext::device() const noexcept
{
    return device_;
}

cl_context OpenClContext::context() const noexcept
{
    return context_;
}

cl_command_queue OpenClContext::create_queue() const
{
    const cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_PROFILING_ENABLE,
        0};
    cl_int error = CL_SUCCESS;
    cl_command_queue queue =
        clCreateCommandQueueWithProperties(
            context_, device_, properties, &error);
    check_cl(error, "clCreateCommandQueueWithProperties");
    return queue;
}

ImportMemoryArm OpenClContext::import_memory() const noexcept
{
    return import_memory_;
}

AcquireExternalMemory
OpenClContext::acquire_external_memory() const noexcept
{
    return acquire_external_memory_;
}

ReleaseExternalMemory
OpenClContext::release_external_memory() const noexcept
{
    return release_external_memory_;
}

OpenClProgramCache& OpenClContext::program_cache() noexcept
{
    return *program_cache_;
}

}  // namespace vision_opencl
