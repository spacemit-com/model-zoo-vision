/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "opencl_error.h"

#include <stdexcept>
#include <string>

namespace vision_opencl {

void check_cl(cl_int error, std::string_view operation)
{
    if (error != CL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed: " +
            std::to_string(error));
    }
}

}  // namespace vision_opencl
