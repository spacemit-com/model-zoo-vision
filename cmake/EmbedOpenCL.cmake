include(CMakeParseArguments)

function(embed_opencl_source)
    cmake_parse_arguments(
        EMBED
        ""
        "OUTPUT;VARIABLE"
        "SOURCES"
        ${ARGN}
    )

    if(NOT EMBED_OUTPUT)
        message(FATAL_ERROR "embed_opencl_source requires OUTPUT")
    endif()
    if(NOT EMBED_VARIABLE)
        message(FATAL_ERROR "embed_opencl_source requires VARIABLE")
    endif()
    if(NOT EMBED_SOURCES)
        message(FATAL_ERROR "embed_opencl_source requires SOURCES")
    endif()

    set(OPENCL_SOURCE "")
    foreach(source_file IN LISTS EMBED_SOURCES)
        if(NOT EXISTS "${source_file}")
            message(FATAL_ERROR
                "OpenCL source does not exist: ${source_file}")
        endif()
        file(READ "${source_file}" source_fragment)
        get_filename_component(source_name "${source_file}" NAME)
        string(APPEND OPENCL_SOURCE
            "// Source: ${source_name}\n"
            "${source_fragment}\n")
    endforeach()

    set(OPENCL_VARIABLE "${EMBED_VARIABLE}")
    get_filename_component(output_directory "${EMBED_OUTPUT}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_directory}")
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/opencl_kernel_source.inc.in"
        "${EMBED_OUTPUT}"
        @ONLY
    )
endfunction()
