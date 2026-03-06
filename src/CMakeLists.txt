cmake_minimum_required(VERSION 3.10)
project(vision_models_cpp VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Build options
option(BUILD_EXAMPLES "Build example programs" ON)
#option(USE_OPENCL "Enable OpenCL support" OFF)

set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv4")
# Find OpenCV
find_package(OpenCV 4 REQUIRED
    NO_DEFAULT_PATH
)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "  Includes: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# ONNX Runtime paths (adjust according to your installation)
if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIR)
    set(ONNXRUNTIME_INCLUDE_DIR "/usr/include")
endif()
if(NOT DEFINED ONNXRUNTIME_LIB_DIR)
    set(ONNXRUNTIME_LIB_DIR "/usr/lib")
endif()

# Check if spacemit_ort_env.h exists
find_path(SPACEMIT_ORT_INCLUDE_DIR
    NAMES spacemit_ort_env.h
    PATHS
        ${ONNXRUNTIME_INCLUDE_DIR}
        /usr/include
        /usr/local/include
    PATH_SUFFIXES onnxruntime
)

if(SPACEMIT_ORT_INCLUDE_DIR)
    message(STATUS "Found spacemit_ort_env.h: ${SPACEMIT_ORT_INCLUDE_DIR}")
else()
    message(WARNING "spacemit_ort_env.h not found, you may need to set SPACEMIT_ORT_INCLUDE_DIR")
endif()

# OpenCL support (optional)
if(USE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_VERSION}")
        add_definitions(-DUSE_OPENCL)
    else()
        message(STATUS "OpenCL not found - building without OpenCL support")
        set(USE_OPENCL OFF)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${SPACEMIT_ORT_INCLUDE_DIR}
    src/deploy/resnet/cpp
    src/deploy/yolov8_seg/cpp
)

if(USE_OPENCL)
    include_directories(${OpenCL_INCLUDE_DIRS})
endif()

# Source files
set(CORE_SOURCES
    core/cpp/vision_model_base.cpp
    core/cpp/vision_model_factory.cpp
)

set(COMMON_SOURCES
    common/cpp/image_processing.cpp
    common/cpp/nms.cpp
    common/cpp/drawing.cpp
    common/cpp/yolo_utils.cpp
    common/cpp/embedding_utils.cpp
)

set(YOLOV8_SOURCES
    deploy/yolov8/cpp/yolov8_detector.cpp
)

set(YOLOV8_POSE_SOURCES
    deploy/yolov8_pose/cpp/yolov8_pose_detector.cpp
)

set(ARC_FACE_SOURCES
    deploy/arcface/cpp/arcface_recognizer.cpp
)

set(YOLOV5_FACE_SOURCES
    deploy/yolov5_face/cpp/yolov5_face_detector.cpp
)

set(RESNET_SOURCES
    deploy/resnet/cpp/resnet_classifier.cpp
)

set(YOLOV8_SEG_SOURCES
    deploy/yolov8_seg/cpp/yolov8_seg_detector.cpp
)

# Create shared library (output: libvision.so)
add_library(vision SHARED
    ${CORE_SOURCES}
    ${COMMON_SOURCES}
    ${YOLOV8_SOURCES}
    ${YOLOV8_POSE_SOURCES}
    ${ARC_FACE_SOURCES}
    ${YOLOV5_FACE_SOURCES}
    ${RESNET_SOURCES}
    ${YOLOV8_SEG_SOURCES}
)

# Link libraries
target_link_libraries(vision
    ${OpenCV_LIBS}
    onnxruntime
    spacemit_ep    
)

if(USE_OPENCL)
    target_link_libraries(vision
        ${OpenCL_LIBRARIES}
        OpenCL::OpenCL
        gbm
    )
endif()

# Link directories
link_directories(${ONNXRUNTIME_LIB_DIR})

# Compiler definitions
add_definitions(-D__fp16=_Float16)

# Build examples
if(BUILD_EXAMPLES)
    add_executable(example_yolov8
        examples/cpp/example_yolov8.cpp
    )
    
    target_link_libraries(example_yolov8
        vision
        ${OpenCV_LIBS}
        onnxruntime
        spacemit_ep        
    )
    
    if(USE_OPENCL)
        target_link_libraries(example_yolov8
            ${OpenCL_LIBRARIES}
            OpenCL::OpenCL
            gbm
        )
    endif()
endif()

# Installation
install(TARGETS vision
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY core/cpp/
    DESTINATION include/vision_models/core
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY common/cpp/
    DESTINATION include/vision_models/common
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY deploy/yolov8/cpp/
    DESTINATION include/vision_models/deploy/yolov8
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY deploy/yolov8_pose/cpp/
    DESTINATION include/vision_models/deploy/yolov8_pose
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY deploy/arcface/cpp/
    DESTINATION include/vision_models/deploy/arcface
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY deploy/yolov5_face/cpp/
    DESTINATION include/vision_models/deploy/yolov5_face
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY deploy/resnet/cpp/
    DESTINATION include/vision_models/deploy/resnet
    FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY deploy/yolov8_seg/cpp/
    DESTINATION include/vision_models/deploy/yolov8_seg
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "")
message(STATUS "==================== CV Models C++ Configuration ====================")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Use OpenCL: ${USE_OPENCL}")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "  ONNX Runtime lib: ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "================================================================")
message(STATUS "")

