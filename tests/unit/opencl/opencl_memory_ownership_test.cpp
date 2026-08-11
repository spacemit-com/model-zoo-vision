/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <string>
#include <type_traits>

#include "backends/opencl/memory/dma_buffer.h"
#include "backends/opencl/memory/dmabuf_import_cache.h"
#include "backends/opencl/memory/external_memory_guard.h"
#include "backends/opencl/memory/opencl_buffer.h"
#include "backends/opencl/runtime/opencl_handles.h"

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
    using vision_opencl::DmaBufIdentity;
    using vision_opencl::DmaBuffer;
    using vision_opencl::ExternalMemoryGuard;
    using vision_opencl::OpenClBuffer;
    using vision_opencl::OpenClCommandQueue;
    using vision_opencl::OpenClKernel;

    static_assert(!std::is_copy_constructible_v<OpenClBuffer>);
    static_assert(std::is_move_constructible_v<OpenClBuffer>);
    static_assert(!std::is_copy_constructible_v<DmaBuffer>);
    static_assert(std::is_move_constructible_v<DmaBuffer>);
    static_assert(!std::is_copy_constructible_v<ExternalMemoryGuard>);
    static_assert(!std::is_copy_constructible_v<OpenClCommandQueue>);
    static_assert(std::is_move_constructible_v<OpenClCommandQueue>);
    static_assert(!std::is_copy_constructible_v<OpenClKernel>);
    static_assert(std::is_move_constructible_v<OpenClKernel>);

    const DmaBufIdentity first{
        1, 2, 1280, 720, 1280, 1280U * 1080U};
    const DmaBufIdentity second{
        1, 2, 1280, 720, 1280, 1280U * 1080U};
    const DmaBufIdentity changed_stride{
        1, 2, 1280, 720, 1344, 1344U * 1080U};

    check(first == second, "equal DMA identities compare equal");
    check(
        !(first == changed_stride),
        "DMA stride participates in import cache identity");

    if (failures != 0) {
        std::cerr << failures << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "PASS: OpenCL memory ownership\n";
    return 0;
}
