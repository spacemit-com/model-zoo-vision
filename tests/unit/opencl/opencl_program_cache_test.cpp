/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <string>

#include "backends/opencl/runtime/opencl_context.h"

namespace {

int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

}  // namespace

int main()
{
    auto context = vision_opencl::OpenClContext::shared();
    constexpr char source[] = "__kernel void noop() {}";
    cl_program first =
        context->program_cache().get_or_build(source, "");
    cl_program second =
        context->program_cache().get_or_build(source, "");

    check(
        first != nullptr,
        "program cache returns a built program");
    check(
        first == second,
        "identical program requests reuse cl_program");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: OpenCL program cache\n";
    return 0;
}
