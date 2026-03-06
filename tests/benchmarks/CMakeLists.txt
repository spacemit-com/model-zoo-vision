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
set(OpenCV_INSTALL_DIR "/home/user/code/test_lib/opencv4x")
#set(OpenCV_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../rvv_libs/opencv-rvv")
set(OpenCV_DIR "${OpenCV_INSTALL_DIR}/lib/cmake/opencv4")
include("${OpenCV_INSTALL_DIR}/lib/cmake/opencv4/OpenCVConfig.cmake")
find_package(OpenCV 4 REQUIRED)

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
