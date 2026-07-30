/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OPENCL_ERROR_H
#define OPENCL_ERROR_H

#include <string_view>

#include <CL/cl.h>

namespace vision_opencl {

void check_cl(cl_int error, std::string_view operation);

}  // namespace vision_opencl

#endif  // OPENCL_ERROR_H
