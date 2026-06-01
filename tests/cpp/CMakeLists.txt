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
        set(OpenCV_DIR "/opt/opencv-spacemit/lib/cmake/opencv4" CACHE PATH "OpenCV config directory")
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

install(TARGETS vision_cpp_functional vision_cpp_invalid_input
    RUNTIME DESTINATION bin
)
