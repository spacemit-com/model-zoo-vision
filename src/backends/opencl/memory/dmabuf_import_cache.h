/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DMABUF_IMPORT_CACHE_H
#define DMABUF_IMPORT_CACHE_H

#include <cstddef>
#include <deque>
#include <memory>
#include <sys/types.h>

#include "opencl_buffer.h"
#include "backends/opencl/runtime/opencl_context.h"
#include "core/cpp/vision_infer_types.h"

namespace vision_opencl {

OpenClBuffer import_dma_buffer(
    const std::shared_ptr<OpenClContext>& context,
    int fd,
    size_t size,
    cl_mem_flags flags = CL_MEM_READ_WRITE);

struct DmaBufIdentity {
    dev_t device{0};
    ino_t inode{0};
    int width{0};
    int height{0};
    int stride{0};
    size_t total_size{0};

    bool operator==(
        const DmaBufIdentity& other) const noexcept;
};

struct ImportedNv12DmaBuffer {
    ~ImportedNv12DmaBuffer();

    int retained_fd{-1};
    DmaBufIdentity identity;
    OpenClBuffer buffer;
    OpenClBuffer y_image;
    OpenClBuffer uv_sub_buffer;
    OpenClBuffer uv_image;
};

class DmaBufImportCache {
public:
    explicit DmaBufImportCache(
        std::shared_ptr<OpenClContext> context,
        size_t capacity = 32);

    ImportedNv12DmaBuffer& get(
        const vision_core::ImageInput& input);

private:
    std::unique_ptr<ImportedNv12DmaBuffer> import(
        const vision_core::ImageInput& input,
        const DmaBufIdentity& identity) const;
    void create_nv12_images(
        ImportedNv12DmaBuffer& input) const;

    std::shared_ptr<OpenClContext> context_;
    size_t capacity_{32};
    std::deque<std::unique_ptr<ImportedNv12DmaBuffer>>
        entries_;
};

}  // namespace vision_opencl

#endif  // DMABUF_IMPORT_CACHE_H
