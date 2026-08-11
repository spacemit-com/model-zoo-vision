cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED VISION_SOURCE_DIR)
    message(FATAL_ERROR "VISION_SOURCE_DIR is required")
endif()
if(NOT DEFINED TEST_OUTPUT_DIR)
    message(FATAL_ERROR "TEST_OUTPUT_DIR is required")
endif()

include("${VISION_SOURCE_DIR}/cmake/EmbedOpenCL.cmake")

file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")
set(test_source "${TEST_OUTPUT_DIR}/sample.cl")
set(test_output "${TEST_OUTPUT_DIR}/sample_kernel.inc")
file(WRITE "${test_source}"
    "__kernel void sample(__global float* value) {\n"
    "    value[get_global_id(0)] *= 2.0f;\n"
    "}\n")

embed_opencl_source(
    OUTPUT "${test_output}"
    VARIABLE "kSampleKernel"
    SOURCES "${test_source}"
)

if(NOT EXISTS "${test_output}")
    message(FATAL_ERROR "embedded output was not generated")
endif()

file(READ "${test_output}" generated)
if(NOT generated MATCHES "kSampleKernel")
    message(FATAL_ERROR "generated output misses the requested variable")
endif()
if(NOT generated MATCHES "__kernel void sample")
    message(FATAL_ERROR "generated output misses the kernel source")
endif()

set(kernel_root
    "${VISION_SOURCE_DIR}/src/backends/opencl/kernels")
set(image_preprocess_output
    "${TEST_OUTPUT_DIR}/image_preprocess_kernel.inc")
embed_opencl_source(
    OUTPUT "${image_preprocess_output}"
    VARIABLE "kImagePreprocessKernelSource"
    SOURCES
        "${kernel_root}/common/color_convert.clh"
        "${kernel_root}/common/sampling.clh"
        "${kernel_root}/common/geometry.clh"
        "${kernel_root}/common/tensor_store.clh"
        "${kernel_root}/image_preprocess.cl"
)
file(READ "${image_preprocess_output}" image_preprocess_generated)
foreach(kernel_name
    preprocess_nv12_images_f32
    preprocess_nv12_images_f16
    preprocess_nv12_images_fast_f32
    preprocess_nv12_images_fast_f16
    preprocess_nv12_images_identity_f32
    preprocess_nv12_images_identity_f16
    preprocess_bgr_buffer_f32
    preprocess_bgr_buffer_f16
    preprocess_bgr_buffer_identity_f32
    preprocess_bgr_buffer_identity_f16
)
    if(NOT image_preprocess_generated MATCHES "${kernel_name}")
        message(FATAL_ERROR
            "generated image preprocess source misses ${kernel_name}")
    endif()
endforeach()
foreach(unsupported_symbol
    preprocess_nv12_buffer_f32
    preprocess_nv12_buffer_f16
    sample_nv12_buffer
    read_bgr_channel
    sample_plane
)
    if(image_preprocess_generated MATCHES "${unsupported_symbol}")
        message(FATAL_ERROR
            "generated image preprocess source unexpectedly contains "
            "${unsupported_symbol}")
    endif()
endforeach()
if(image_preprocess_generated MATCHES "${VISION_SOURCE_DIR}")
    message(FATAL_ERROR
        "generated OpenCL source contains an absolute checkout path")
endif()
