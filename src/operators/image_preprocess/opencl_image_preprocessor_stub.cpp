/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "image_preprocessor.h"

#include <stdexcept>

namespace vision_operators {

std::shared_ptr<ImagePreprocessor>
create_opencl_image_preprocessor(
    const ImagePreprocessSpec&,
    int)
{
    throw std::runtime_error(
        "OpenCL image preprocessing was not compiled");
}

bool opencl_image_preprocessor_compiled() noexcept
{
    return false;
}

}  // namespace vision_operators
