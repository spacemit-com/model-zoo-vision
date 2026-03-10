cmake_minimum_required(VERSION 3.10)

# Vision root: parent of tests/benchmarks (works when built from cv/build or tests/benchmarks/build).
set(VISION_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# If vision target is not present, build it from vision root.
if(NOT TARGET vision)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${VISION_ROOT} ${CMAKE_BINARY_DIR}/vision_build)
endif()

# Link libvision.so (shared; self-registration is included in the .so).
function(link_vision_with_registration target_name)
    target_link_libraries(${target_name} PRIVATE vision)
endfunction()

add_executable(vision_infer_benchmark infer_benchmark.cpp)

# OpenCV Configuration
if(NOT OpenCV_FOUND)
    if(NOT DEFINED OpenCV_DIR)
        set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv4" CACHE PATH "OpenCV config directory")
    endif()
    find_package(OpenCV 4 REQUIRED NO_DEFAULT_PATH)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui NO_DEFAULT_PATH)
endif()

find_package(OpenCV REQUIRED
    COMPONENTS core imgproc highgui
    NO_DEFAULT_PATH
)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "  Includes: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found! Check OpenCV_DIR or install OpenCV.")
endif()

set(LIBS 
    ${OpenCV_LIBS} 
    onnxruntime 
    spacemit_ep     
    yaml-cpp
)
link_vision_with_registration(vision_infer_benchmark)
target_link_libraries(vision_infer_benchmark PRIVATE ${LIBS})

# ONNX Runtime float16: same as vision target (x86 may lack native __fp16)
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_QUIET TRUE)
check_cxx_source_compiles(
    "int main() { __fp16 x = 0; (void)x; return 0; }"
    HAVE_NATIVE_FP16_BENCHMARK
)
unset(CMAKE_REQUIRED_QUIET)
if(NOT HAVE_NATIVE_FP16_BENCHMARK)
    target_compile_definitions(vision_infer_benchmark PRIVATE __fp16=_Float16)
    message(STATUS "vision_infer_benchmark: defining __fp16=_Float16 for ONNX Runtime headers")
endif()


target_include_directories(vision_infer_benchmark PRIVATE
    ${VISION_ROOT}/include
    ${VISION_ROOT}/src
    ${VISION_ROOT}/src/core/cpp
    ${VISION_ROOT}/src/common/cpp
    ${OpenCV_INCLUDE_DIRS}
)
if(DEFINED yaml-cpp_INCLUDE_DIR)
    target_include_directories(vision_infer_benchmark PRIVATE ${yaml-cpp_INCLUDE_DIR})
endif()
if(DEFINED SPACEMIT_DIR)
    target_include_directories(vision_infer_benchmark PRIVATE ${SPACEMIT_DIR}/include)
endif()
