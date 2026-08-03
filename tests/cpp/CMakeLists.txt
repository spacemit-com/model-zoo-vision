cmake_minimum_required(VERSION 3.10)

set(VISION_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

if(NOT TARGET vision)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${VISION_ROOT} ${CMAKE_BINARY_DIR}/vision_build)
endif()

function(link_vision_test target_name)
    target_link_libraries(${target_name} PRIVATE vision)
endfunction()

if(NOT OpenCV_FOUND)
    if(NOT DEFINED OpenCV_DIR)
        if(EXISTS "/opt/opencv-spacemit/lib/cmake/opencv5/OpenCVConfig.cmake")
            set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv5" CACHE PATH "OpenCV config directory")
        elseif(EXISTS "/opt/opencv-spacemit/lib/cmake/opencv4/OpenCVConfig.cmake")
            set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv4" CACHE PATH "OpenCV config directory")
        else()
            set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv5" CACHE PATH "OpenCV config directory")
        endif()
    endif()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui NO_DEFAULT_PATH)
endif()

if(NOT yaml-cpp_FOUND)
    find_package(yaml-cpp REQUIRED)
endif()

set(VISION_TEST_LIBS
    ${OpenCV_LIBS}
    onnxruntime
    spacemit_ep
    yaml-cpp
)

macro(configure_vision_cpp_test target_name source_file)
    add_executable(${target_name} ${source_file})
    link_vision_test(${target_name})
    target_link_libraries(${target_name} PRIVATE ${VISION_TEST_LIBS})

    set(_vision_cpp_test_rpath "\$ORIGIN;\$ORIGIN/..;\$ORIGIN/../..")
    if(DEFINED OpenCV_DIR)
        get_filename_component(_opencv_install_libdir "${OpenCV_DIR}/../.." ABSOLUTE)
        list(APPEND _vision_cpp_test_rpath "${_opencv_install_libdir}")
    endif()
    if(DEFINED SPACEMIT_DIR)
        list(APPEND _vision_cpp_test_rpath "${SPACEMIT_DIR}/lib")
    endif()
    set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "${CMAKE_BUILD_RPATH}"
        INSTALL_RPATH "${_vision_cpp_test_rpath}"
    )

    target_include_directories(${target_name} PRIVATE
        ${VISION_ROOT}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    if(DEFINED yaml-cpp_INCLUDE_DIR)
        target_include_directories(${target_name} PRIVATE ${yaml-cpp_INCLUDE_DIR})
    endif()
    if(DEFINED SPACEMIT_DIR)
        target_include_directories(${target_name} PRIVATE ${SPACEMIT_DIR}/include)
    endif()
endmacro()

configure_vision_cpp_test(vision_cpp_functional cpp_functional.cpp)
configure_vision_cpp_test(vision_cpp_invalid_input cpp_invalid_input.cpp)
configure_vision_cpp_test(vision_cpp_runtime_profile cpp_runtime_profile.cpp)
configure_vision_cpp_test(
    vision_extended_infer_contract_test
    extended_infer_contract_test.cpp
)
configure_vision_cpp_test(
    vision_banet2d_postprocess_test
    banet2d_postprocess_test.cpp
)
configure_vision_cpp_test(
    vision_local_feature_models_test
    local_feature_models_test.cpp
)
configure_vision_cpp_test(
    vision_single_object_tracker_contract_test
    single_object_tracker_contract_test.cpp
)
configure_vision_cpp_test(
    vision_preprocess_backend_scope_test
    preprocess_backend_scope_test.cpp
)
target_include_directories(vision_extended_infer_contract_test PRIVATE
    ${VISION_ROOT}/src
    ${VISION_ROOT}/src/core/cpp
    ${VISION_ROOT}/src/common/cpp
)
target_include_directories(vision_banet2d_postprocess_test PRIVATE
    ${VISION_ROOT}/src/deploy/banet2d/cpp
    ${VISION_ROOT}/src
    ${VISION_ROOT}/src/core/cpp
    ${VISION_ROOT}/src/common/cpp
)
target_include_directories(vision_local_feature_models_test PRIVATE
    ${VISION_ROOT}/src
    ${VISION_ROOT}/src/core/cpp
    ${VISION_ROOT}/src/common/cpp
    ${VISION_ROOT}/src/deploy/superpoint/cpp
    ${VISION_ROOT}/src/deploy/lightglue/cpp
)
target_include_directories(vision_single_object_tracker_contract_test PRIVATE
    ${VISION_ROOT}/src
    ${VISION_ROOT}/src/core/cpp
    ${VISION_ROOT}/src/common/cpp
    ${VISION_ROOT}/src/deploy/mixformer/cpp
    ${VISION_ROOT}/src/deploy/avtrack/cpp
    ${VISION_ROOT}/src/deploy/nanotrack/cpp
)
add_test(
    NAME vision_extended_infer_contract
    COMMAND vision_extended_infer_contract_test
)
add_test(
    NAME vision_banet2d_postprocess
    COMMAND vision_banet2d_postprocess_test
)
add_test(
    NAME vision_local_feature_models
    COMMAND vision_local_feature_models_test
)
add_test(
    NAME vision_single_object_tracker_contract
    COMMAND vision_single_object_tracker_contract_test
)
target_include_directories(vision_preprocess_backend_scope_test PRIVATE
    ${VISION_ROOT}/src
    ${VISION_ROOT}/src/core/cpp
    ${VISION_ROOT}/src/common/cpp
)
target_compile_definitions(vision_preprocess_backend_scope_test PRIVATE
    VISION_WITH_OPENCL=$<BOOL:${VISION_WITH_OPENCL}>
)

configure_vision_cpp_test(
    vision_mpp_frame_api_test
    mpp_frame_source_api_test.cpp
)
target_include_directories(vision_mpp_frame_api_test PRIVATE
    ${VISION_ROOT}/src/common/cpp
)
add_test(
    NAME vision_mpp_frame_api
    COMMAND vision_mpp_frame_api_test
)

add_executable(
    vision_mpp_nv12_layout_test
    mpp_nv12_layout_test.cpp
    ${VISION_ROOT}/src/common/cpp/mpp_nv12_layout.cpp
)
target_include_directories(vision_mpp_nv12_layout_test PRIVATE
    ${VISION_ROOT}/src/common/cpp
)
add_test(
    NAME vision_mpp_nv12_layout
    COMMAND vision_mpp_nv12_layout_test
)

add_executable(
    vision_opencl_geometry_test
    opencl/image_preprocess_geometry_test.cpp
    ${VISION_ROOT}/src/operators/image_preprocess/image_preprocess_geometry.cpp
)
target_include_directories(vision_opencl_geometry_test PRIVATE
    ${VISION_ROOT}/src
)
add_test(
    NAME vision_opencl_geometry
    COMMAND vision_opencl_geometry_test
)

add_executable(
    vision_preprocess_backend_test
    opencl/image_preprocess_backend_test.cpp
    ${VISION_ROOT}/src/operators/image_preprocess/image_preprocess_backend.cpp
)
target_include_directories(vision_preprocess_backend_test PRIVATE
    ${VISION_ROOT}/src
)
add_test(
    NAME vision_preprocess_backend
    COMMAND vision_preprocess_backend_test
)

configure_vision_cpp_test(
    vision_preprocess_dispatcher_test
    opencl/image_preprocess_dispatcher_test.cpp
)
target_include_directories(vision_preprocess_dispatcher_test PRIVATE
    ${VISION_ROOT}/src
)
target_compile_definitions(vision_preprocess_dispatcher_test PRIVATE
    VISION_WITH_OPENCL=$<BOOL:${VISION_WITH_OPENCL}>
)
add_test(
    NAME vision_preprocess_dispatcher
    COMMAND vision_preprocess_dispatcher_test
)

configure_vision_cpp_test(
    vision_cpu_image_preprocessor_test
    opencl/cpu_image_preprocessor_test.cpp
)
target_include_directories(vision_cpu_image_preprocessor_test PRIVATE
    ${VISION_ROOT}/src
)
add_test(
    NAME vision_cpu_image_preprocessor
    COMMAND vision_cpu_image_preprocessor_test
)

add_test(
    NAME vision_embed_opencl_source
    COMMAND
        ${CMAKE_COMMAND}
        -DVISION_SOURCE_DIR=${VISION_ROOT}
        -DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/embed_opencl_test
        -P ${VISION_ROOT}/tests/cmake/embed_opencl_test.cmake
)

add_test(
    NAME vision_opencl_optional_build
    COMMAND
        ${CMAKE_COMMAND}
        -DVISION_SOURCE_DIR=${VISION_ROOT}
        -DTEST_OUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/opencl_optional_build
        -DOpenCV_DIR=${OpenCV_DIR}
        -DSPACEMIT_DIR=${SPACEMIT_DIR}
        -P ${VISION_ROOT}/tests/cmake/opencl_optional_build_test.cmake
)

if(VISION_WITH_OPENCL)
    configure_vision_cpp_test(
        vision_opencl_image_preprocess_golden_test
        opencl/opencl_image_preprocess_golden_test.cpp
    )
    target_include_directories(
        vision_opencl_image_preprocess_golden_test
        PRIVATE ${VISION_ROOT}/src
    )
    add_test(
        NAME vision_opencl_image_preprocess_golden
        COMMAND vision_opencl_image_preprocess_golden_test
    )

    add_executable(
        vision_opencl_program_cache_test
        opencl/opencl_program_cache_test.cpp
    )
    target_link_libraries(
        vision_opencl_program_cache_test
        PRIVATE vision_opencl_backend
    )
    target_include_directories(
        vision_opencl_program_cache_test
        PRIVATE ${VISION_ROOT}/src
    )
    add_test(
        NAME vision_opencl_program_cache
        COMMAND vision_opencl_program_cache_test
    )

    add_executable(
        vision_opencl_memory_ownership_test
        opencl/opencl_memory_ownership_test.cpp
    )
    target_link_libraries(
        vision_opencl_memory_ownership_test
        PRIVATE vision_opencl_backend
    )
    target_include_directories(
        vision_opencl_memory_ownership_test
        PRIVATE
            ${VISION_ROOT}/include
            ${VISION_ROOT}/src
            ${VISION_ROOT}/src/core/cpp
            ${VISION_ROOT}/src/common/cpp
            ${OpenCV_INCLUDE_DIRS}
    )
    add_test(
        NAME vision_opencl_memory_ownership
        COMMAND vision_opencl_memory_ownership_test
    )
endif()

target_include_directories(vision_cpp_runtime_profile PRIVATE
    ${VISION_ROOT}/src
    ${VISION_ROOT}/src/core/cpp
    ${VISION_ROOT}/src/common/cpp
)

add_test(NAME vision_cpp_runtime_profile COMMAND vision_cpp_runtime_profile)
add_test(
    NAME vision_preprocess_backend_scope
    COMMAND vision_preprocess_backend_scope_test
)

install(TARGETS
    vision_cpp_functional
    vision_cpp_invalid_input
    vision_cpp_runtime_profile
    RUNTIME DESTINATION bin
)
