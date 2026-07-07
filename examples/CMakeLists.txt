cmake_minimum_required(VERSION 3.10)

# Link libvision.so (shared; self-registration is included in the .so).
function(link_vision_with_registration target_name)
    target_link_libraries(${target_name} PRIVATE vision)
endfunction()

# Initialize example targets list
set(EXAMPLE_TARGETS)

# Examples use only vision_service.h API. Discover examples from the
# conventional examples/<name>/cpp/<name>.cpp layout so new model examples do
# not require a CMake edit.
file(GLOB EXAMPLE_CPP_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*/cpp/*.cpp"
)

foreach(cpp_file IN LISTS EXAMPLE_CPP_FILES)
    get_filename_component(example_cpp_dir "${cpp_file}" DIRECTORY)
    get_filename_component(example_dir "${example_cpp_dir}" DIRECTORY)
    get_filename_component(example_name "${example_dir}" NAME)
    get_filename_component(example_stem "${cpp_file}" NAME_WE)

    if(NOT example_name STREQUAL example_stem)
        message(STATUS "Skipping example source with non-conventional name: ${cpp_file}")
        continue()
    endif()

    add_executable(${example_name} ${cpp_file})
    link_vision_with_registration(${example_name})
    target_link_libraries(${example_name} PRIVATE ${LIBS})

    target_include_directories(${example_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/common/cpp
    )

    if(VISION_WITH_MPP)
        target_compile_definitions(${example_name} PRIVATE VISION_HAS_MPP=1)
        set_property(TARGET ${example_name} APPEND PROPERTY BUILD_RPATH "${VISION_MPP_RPATH}")
        set_property(TARGET ${example_name} APPEND PROPERTY INSTALL_RPATH "${VISION_MPP_RPATH}")
    endif()

    list(APPEND EXAMPLE_TARGETS ${example_name})
endforeach()

# Install examples (only install targets that were actually created)
if(EXAMPLE_TARGETS)
    install(TARGETS ${EXAMPLE_TARGETS} DESTINATION bin)
endif()