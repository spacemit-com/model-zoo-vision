/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "opencl_program_cache.h"

#include <stdexcept>
#include <string>

#include "opencl_error.h"

namespace vision_opencl {

OpenClProgramCache::OpenClProgramCache(
    cl_context context,
    cl_device_id device)
    : context_(context),
        device_(device)
{
}

OpenClProgramCache::~OpenClProgramCache()
{
    for (const auto& entry : programs_) {
        if (entry.second != nullptr) {
            clReleaseProgram(entry.second);
        }
    }
}

cl_program OpenClProgramCache::get_or_build(
    std::string_view source,
    std::string_view build_options)
{
    std::string key;
    key.reserve(source.size() + build_options.size() + 1U);
    key.append(build_options.data(), build_options.size());
    key.push_back('\0');
    key.append(source.data(), source.size());

    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = programs_.find(key);
    if (found != programs_.end()) {
        return found->second;
    }

    cl_program program = build(source, build_options);
    programs_.emplace(std::move(key), program);
    return program;
}

cl_program OpenClProgramCache::build(
    std::string_view source,
    std::string_view build_options) const
{
    const char* source_data = source.data();
    const size_t source_size = source.size();
    cl_int error = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(
        context_, 1, &source_data, &source_size, &error);
    check_cl(error, "clCreateProgramWithSource");

    const std::string options(build_options);
    error = clBuildProgram(
        program,
        1,
        &device_,
        options.empty() ? nullptr : options.c_str(),
        nullptr,
        nullptr);
    if (error == CL_SUCCESS) {
        return program;
    }

    size_t log_size = 0;
    clGetProgramBuildInfo(
        program,
        device_,
        CL_PROGRAM_BUILD_LOG,
        0,
        nullptr,
        &log_size);
    std::string log(log_size, '\0');
    if (log_size > 0) {
        clGetProgramBuildInfo(
            program,
            device_,
            CL_PROGRAM_BUILD_LOG,
            log.size(),
            log.data(),
            nullptr);
    }
    clReleaseProgram(program);
    throw std::runtime_error(
        "OpenCL program build failed: " + log);
}

}  // namespace vision_opencl
