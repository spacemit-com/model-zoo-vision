find_package(OpenCL REQUIRED)

include("${PROJECT_SOURCE_DIR}/cmake/EmbedOpenCL.cmake")

set(OPENCL_IMAGE_PREPROCESS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/color_convert.clh"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/sampling.clh"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/geometry.clh"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/common/tensor_store.clh"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/image_preprocess.cl"
)
set(OPENCL_IMAGE_PREPROCESS_GENERATED
    "${PROJECT_BINARY_DIR}/generated/opencl/image_preprocess_kernel.inc"
)
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${OPENCL_IMAGE_PREPROCESS_SOURCES}
)
embed_opencl_source(
    OUTPUT "${OPENCL_IMAGE_PREPROCESS_GENERATED}"
    VARIABLE "kImagePreprocessKernelSource"
    SOURCES ${OPENCL_IMAGE_PREPROCESS_SOURCES}
)

add_library(vision_opencl_backend STATIC
    runtime/opencl_error.cpp
    runtime/opencl_handles.cpp
    runtime/opencl_context.cpp
    runtime/opencl_program_cache.cpp
    memory/opencl_buffer.cpp
    memory/dma_buffer.cpp
    memory/dmabuf_import_cache.cpp
    memory/external_memory_guard.cpp
    operators/opencl_image_preprocessor.cpp
    "${OPENCL_IMAGE_PREPROCESS_GENERATED}"
)
set_target_properties(vision_opencl_backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Silence cl_version.h pragma when CL_TARGET_OPENCL_VERSION is unset.
# PUBLIC so dependents that include <CL/cl.h> via our headers also get it.
target_compile_definitions(vision_opencl_backend
    PUBLIC CL_TARGET_OPENCL_VERSION=300
)

target_link_libraries(vision_opencl_backend
    PRIVATE
        OpenCL::OpenCL
        ${OpenCV_LIBS}
)
target_include_directories(vision_opencl_backend
    PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src"
        "${PROJECT_SOURCE_DIR}/src/core/cpp"
        "${PROJECT_SOURCE_DIR}/src/common/cpp"
        "${PROJECT_BINARY_DIR}"
        ${OpenCV_INCLUDE_DIRS}
)
