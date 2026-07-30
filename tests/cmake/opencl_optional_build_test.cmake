cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED VISION_SOURCE_DIR)
    message(FATAL_ERROR "VISION_SOURCE_DIR is required")
endif()
if(NOT DEFINED TEST_OUTPUT_DIR)
    message(FATAL_ERROR "TEST_OUTPUT_DIR is required")
endif()

file(REMOVE_RECURSE "${TEST_OUTPUT_DIR}")

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${VISION_SOURCE_DIR}"
    -B "${TEST_OUTPUT_DIR}"
    -DVISION_WITH_OPENCL=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_PYTHON_BINDINGS=OFF
    -DBUILD_PYTHON_WHEEL=OFF
)
if(DEFINED OpenCV_DIR AND NOT OpenCV_DIR STREQUAL "")
    list(APPEND configure_command "-DOpenCV_DIR=${OpenCV_DIR}")
endif()
if(DEFINED SPACEMIT_DIR AND NOT SPACEMIT_DIR STREQUAL "")
    list(APPEND configure_command "-DSPACEMIT_DIR=${SPACEMIT_DIR}")
endif()

execute_process(
    COMMAND ${configure_command}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error
)

if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
        "CPU-only configure failed:\n"
        "${configure_output}\n${configure_error}")
endif()

file(READ "${TEST_OUTPUT_DIR}/CMakeCache.txt" cache)
if(cache MATCHES
        "OpenCL_LIBRARY:FILEPATH=[^\n]+")
    message(FATAL_ERROR
        "VISION_WITH_OPENCL=OFF still resolves OpenCL_LIBRARY")
endif()
