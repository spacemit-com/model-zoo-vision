/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENCL_PROGRAM_CACHE_H
#define OPENCL_PROGRAM_CACHE_H

#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include <CL/cl.h>

namespace vision_opencl {

class OpenClProgramCache {
public:
    OpenClProgramCache(
        cl_context context,
        cl_device_id device);
    ~OpenClProgramCache();

    OpenClProgramCache(const OpenClProgramCache&) = delete;
    OpenClProgramCache& operator=(
        const OpenClProgramCache&) = delete;

    cl_program get_or_build(
        std::string_view source,
        std::string_view build_options);

private:
    cl_program build(
        std::string_view source,
        std::string_view build_options) const;

    cl_context context_{nullptr};
    cl_device_id device_{nullptr};
    std::mutex mutex_;
    std::unordered_map<std::string, cl_program> programs_;
};

}  // namespace vision_opencl

#endif  // OPENCL_PROGRAM_CACHE_H
