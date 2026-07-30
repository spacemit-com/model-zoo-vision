/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dmabuf_import_cache.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include "backends/opencl/runtime/opencl_error.h"

namespace vision_opencl {

OpenClBuffer import_dma_buffer(
    const std::shared_ptr<OpenClContext>& context,
    int fd,
    size_t size,
    cl_mem_flags flags)
{
    if (!context || fd < 0 || size == 0) {
        throw std::invalid_argument(
            "DMA-BUF import requires context, fd, and size");
    }
    const cl_import_properties_arm properties[] = {
        CL_IMPORT_TYPE_ARM,
        CL_IMPORT_TYPE_DMA_BUF_ARM,
        0};
    cl_int error = CL_SUCCESS;
    cl_mem memory = context->import_memory()(
        context->context(),
        flags,
        properties,
        &fd,
        size,
        &error);
    check_cl(error, "clImportMemoryARM");
    return OpenClBuffer(memory);
}

bool DmaBufIdentity::operator==(
    const DmaBufIdentity& other) const noexcept
{
    return device == other.device &&
        inode == other.inode &&
        width == other.width &&
        height == other.height &&
        stride == other.stride &&
        total_size == other.total_size;
}

ImportedNv12DmaBuffer::~ImportedNv12DmaBuffer()
{
    uv_image.reset();
    uv_sub_buffer.reset();
    y_image.reset();
    buffer.reset();
    if (retained_fd >= 0) {
        ::close(retained_fd);
    }
}

DmaBufImportCache::DmaBufImportCache(
    std::shared_ptr<OpenClContext> context,
    size_t capacity)
    : context_(std::move(context)),
        capacity_(capacity)
{
    if (!context_) {
        throw std::invalid_argument(
            "OpenCL context is required");
    }
    if (capacity_ == 0) {
        throw std::invalid_argument(
            "DMA-BUF import cache capacity must be positive");
    }
}

ImportedNv12DmaBuffer& DmaBufImportCache::get(
    const vision_core::ImageInput& input)
{
    if (input.format !=
            vision_core::ImagePixelFormat::kNv12 ||
        input.dma_fd < 0) {
        throw std::invalid_argument(
            "OpenCL image preprocessing requires "
            "NV12 DMA-BUF input");
    }

    struct stat info {};
    if (::fstat(input.dma_fd, &info) != 0) {
        throw std::invalid_argument(
            "fstat(input dma-buf) failed: " +
            std::string(std::strerror(errno)));
    }
    const DmaBufIdentity identity{
        info.st_dev,
        info.st_ino,
        input.image.cols,
        input.image.rows * 2 / 3,
        static_cast<int>(input.image.step[0]),
        static_cast<size_t>(input.image.step[0]) *
            input.image.rows};

    for (auto iterator = entries_.begin();
        iterator != entries_.end();
        ++iterator) {
        if ((*iterator)->identity == identity) {
            if (iterator != entries_.begin()) {
                auto entry = std::move(*iterator);
                entries_.erase(iterator);
                entries_.push_front(std::move(entry));
            }
            return *entries_.front();
        }
    }

    entries_.push_front(import(input, identity));
    if (entries_.size() > capacity_) {
        entries_.pop_back();
    }
    return *entries_.front();
}

std::unique_ptr<ImportedNv12DmaBuffer>
DmaBufImportCache::import(
    const vision_core::ImageInput& input,
    const DmaBufIdentity& identity) const
{
    auto imported =
        std::make_unique<ImportedNv12DmaBuffer>();
    imported->retained_fd = ::dup(input.dma_fd);
    if (imported->retained_fd < 0) {
        throw std::runtime_error(
            "dup(input dma-buf) failed: " +
            std::string(std::strerror(errno)));
    }
    imported->identity = identity;
    imported->buffer = import_dma_buffer(
        context_,
        imported->retained_fd,
        identity.total_size,
        CL_MEM_READ_ONLY);
    create_nv12_images(*imported);
    return imported;
}

void DmaBufImportCache::create_nv12_images(
    ImportedNv12DmaBuffer& input) const
{
    cl_image_format y_format{};
    y_format.image_channel_order = CL_R;
    y_format.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc y_description{};
    y_description.image_type = CL_MEM_OBJECT_IMAGE2D;
    y_description.image_width =
        static_cast<size_t>(input.identity.width);
    y_description.image_height =
        static_cast<size_t>(input.identity.height);
    y_description.image_row_pitch =
        static_cast<size_t>(input.identity.stride);
    y_description.buffer = input.buffer.get();

    cl_int error = CL_SUCCESS;
    input.y_image.reset(clCreateImage(
        context_->context(),
        CL_MEM_READ_ONLY,
        &y_format,
        &y_description,
        nullptr,
        &error));
    check_cl(error, "clCreateImage(Y)");

    const size_t uv_offset =
        static_cast<size_t>(input.identity.stride) *
        input.identity.height;
    if (uv_offset >= input.identity.total_size) {
        throw std::invalid_argument(
            "NV12 DMA-BUF has no UV plane");
    }
    cl_buffer_region uv_region{};
    uv_region.origin = uv_offset;
    uv_region.size =
        input.identity.total_size - uv_offset;
    input.uv_sub_buffer.reset(clCreateSubBuffer(
        input.buffer.get(),
        CL_MEM_READ_ONLY,
        CL_BUFFER_CREATE_TYPE_REGION,
        &uv_region,
        &error));
    check_cl(error, "clCreateSubBuffer(UV)");

    cl_image_format uv_format{};
    uv_format.image_channel_order = CL_RG;
    uv_format.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc uv_description{};
    uv_description.image_type = CL_MEM_OBJECT_IMAGE2D;
    uv_description.image_width =
        static_cast<size_t>(input.identity.width / 2);
    uv_description.image_height =
        static_cast<size_t>(input.identity.height / 2);
    uv_description.image_row_pitch =
        static_cast<size_t>(input.identity.stride);
    uv_description.buffer = input.uv_sub_buffer.get();
    input.uv_image.reset(clCreateImage(
        context_->context(),
        CL_MEM_READ_ONLY,
        &uv_format,
        &uv_description,
        nullptr,
        &error));
    check_cl(error, "clCreateImage(UV)");
}

}  // namespace vision_opencl
