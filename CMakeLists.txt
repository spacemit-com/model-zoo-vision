cmake_minimum_required(VERSION 3.10)
project(ModelZoo VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# Project Configuration
# ============================================================================

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# RPATH / Runtime Library Search Paths
# ============================================================================
# Make binaries runnable from both build/ and installed (staging) layouts.
# - Build tree: use link paths so local runs find OpenCV/onnxruntime/spacemit libs.
# - Install tree: search libs relative to the binary (../lib).
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

set(_modelzoo_build_rpath "")
if(DEFINED OpenCV_INSTALL_DIR)
    list(APPEND _modelzoo_build_rpath "${OpenCV_INSTALL_DIR}/lib")
endif()
if(DEFINED SPACEMIT_DIR)
    list(APPEND _modelzoo_build_rpath "${SPACEMIT_DIR}/lib")
endif()
set(CMAKE_BUILD_RPATH "${_modelzoo_build_rpath}")
set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib:\$ORIGIN")

# Build Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests" OFF)

# ============================================================================
# Dependencies Configuration
# ============================================================================

set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv4")
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


# Eigen3 (system)
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
endif()

# yaml-cpp Configuration
find_package(yaml-cpp REQUIRED)
if(yaml-cpp_FOUND)
    message(STATUS "yaml-cpp found: ${yaml-cpp_VERSION}")
    message(STATUS "  Includes: ${yaml-cpp_INCLUDE_DIR}")
    message(STATUS "  Libraries: ${yaml-cpp_LIBRARIES}")
else()
    message(FATAL_ERROR "yaml-cpp not found! Please install yaml-cpp.")
endif()

# Library Dependencies
set(LIBS 
    ${OpenCV_LIBS} 
    onnxruntime 
    spacemit_ep     
    yaml-cpp
)

# ============================================================================
# Source Files
# ============================================================================

file(GLOB_RECURSE VISION_CORE_SOURCES "src/core/cpp/*.cpp")
set(VISION_SERVICE_SOURCES
    src/vision_service.cpp
)
file(GLOB_RECURSE CV_COMMON_SOURCES "src/common/cpp/*.cpp")
file(GLOB_RECURSE CV_YOLOV8_SOURCES "src/deploy/yolov8/cpp/*.cpp")
file(GLOB_RECURSE CV_YOLOV8_POSE_SOURCES "src/deploy/yolov8_pose/cpp/*.cpp")
file(GLOB_RECURSE CV_ARCFACE_SOURCES "src/deploy/arcface/cpp/*.cpp")
file(GLOB_RECURSE CV_YOLOV5_FACE_SOURCES "src/deploy/yolov5_face/cpp/*.cpp")
file(GLOB_RECURSE CV_YOLOV5_GESTURE_SOURCES "src/deploy/yolov5_gesture/cpp/*.cpp")
file(GLOB_RECURSE CV_RESNET_SOURCES "src/deploy/resnet/cpp/*.cpp")
file(GLOB_RECURSE CV_YOLOV8_SEG_SOURCES "src/deploy/yolov8_seg/cpp/*.cpp")
file(GLOB_RECURSE CV_EMOTION_SOURCES "src/deploy/emotion/cpp/*.cpp")
file(GLOB_RECURSE CV_STGCN_SOURCES "src/deploy/stgcn/cpp/*.cpp")
file(GLOB_RECURSE CV_BYTETRACK_SOURCES "src/deploy/bytetrack/cpp/src/*.cpp")
file(GLOB_RECURSE CV_BYTETRACK_TRACKER_SOURCES "src/deploy/bytetrack/cpp/byte_track_tracker.cpp")
file(GLOB_RECURSE CV_OCSORT_SOURCES "src/deploy/ocsort/cpp/src/*.cpp")
file(GLOB_RECURSE CV_OCSORT_TRACKER_SOURCES "src/deploy/ocsort/cpp/ocsort_tracker.cpp")

# ============================================================================
# Library Target (output: libvision.so)
# ============================================================================

add_library(vision SHARED
    ${VISION_SERVICE_SOURCES}
    ${VISION_CORE_SOURCES}
    ${CV_COMMON_SOURCES}
    ${CV_YOLOV8_SOURCES}
    ${CV_YOLOV8_POSE_SOURCES}
    ${CV_ARCFACE_SOURCES}
    ${CV_YOLOV5_FACE_SOURCES}
    ${CV_YOLOV5_GESTURE_SOURCES}
    ${CV_RESNET_SOURCES}
    ${CV_YOLOV8_SEG_SOURCES}
    ${CV_EMOTION_SOURCES}
    ${CV_STGCN_SOURCES}
    ${CV_BYTETRACK_SOURCES}
    ${CV_BYTETRACK_TRACKER_SOURCES}
    ${CV_OCSORT_SOURCES}
    ${CV_OCSORT_TRACKER_SOURCES}
)

target_link_libraries(vision
    ${LIBS}
)

target_include_directories(vision PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/yolov8/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/yolov8_pose/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/arcface/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/yolov5_face/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/yolov5_gesture/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/resnet/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/yolov8_seg/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/emotion/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/stgcn/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/bytetrack/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/bytetrack/cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deploy/ocsort/cpp
    Eigen3::Eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${OpenCV_INCLUDE_DIRS}
    ${SPACEMIT_DIR}/include
    ${yaml-cpp_INCLUDE_DIR}
)

# Public API: only vision_service.h is exposed; internal headers are not installed.
target_include_directories(vision PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ============================================================================
# Subdirectories
# ============================================================================

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Applications (applications/<name>/cpp/ and config/)
if(BUILD_EXAMPLES)
    add_subdirectory(applications)
endif()

# Tests and benchmarks
if(BUILD_TESTS)
    enable_testing()    
    add_subdirectory(tests/benchmarks)
endif()

# ============================================================================
# Installation (public API: library + vision_service.h only)
# ============================================================================

install(TARGETS vision
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Only expose vision_service.h; internal headers are not installed.
install(FILES include/vision_service.h
    DESTINATION include
)

# ============================================================================
# Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "==================== ModelZoo Configuration ====================")
message(STATUS "  Version:              ${PROJECT_VERSION}")
message(STATUS "  C++ Compiler:         ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build examples:       ${BUILD_EXAMPLES}")
message(STATUS "  Build tests:           ${BUILD_TESTS}")
message(STATUS "  OpenCV version:        ${OpenCV_VERSION}")
message(STATUS "  OpenCV libs:           ${OpenCV_LIBS}")
message(STATUS "  SpaceMIT dir:         ${SPACEMIT_DIR}")
message(STATUS "================================================================")
message(STATUS "")
