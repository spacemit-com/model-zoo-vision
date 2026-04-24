# Applications: each app lives under applications/<app_name>/cpp/ and config/
# Link libvision.so (shared; self-registration is included in the .so).
function(link_vision_with_registration target_name)
    target_link_libraries(${target_name} PRIVATE vision)
endfunction()

# Add one application executable.
# Usage:
#   add_application(example_emotion emotion_detection/cpp/example_emotion.cpp yolov5_face emotion)
#   add_application(example_fall_detection fall_detection/cpp/example_fall_detection.cpp yolov8_pose)
#   add_application(example_intrusion ... bytetrack EXTRA_INCLUDES /path/to/eigen)
function(add_application target_name cpp_file)
    set(extra_includes "")
    set(deploy_dirs "")
    set(doing_extra OFF)
    foreach(arg IN LISTS ARGN)
        if(arg STREQUAL "EXTRA_INCLUDES")
            set(doing_extra ON)
        elseif(doing_extra)
            list(APPEND extra_includes ${arg})
        else()
            list(APPEND deploy_dirs ${arg})
        endif()
    endforeach()

    set(cpp_path "${CMAKE_CURRENT_SOURCE_DIR}/${cpp_file}")
    if(NOT EXISTS "${cpp_path}")
        message(STATUS "Application source not found, skipping: ${cpp_path}")
        return()
    endif()

    add_executable(${target_name} ${cpp_file})
    link_vision_with_registration(${target_name})
    target_link_libraries(${target_name} PRIVATE ${LIBS})

    set(all_includes
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core/cpp
        ${CMAKE_SOURCE_DIR}/src/common/cpp
    )
    foreach(dep IN LISTS deploy_dirs)
        list(APPEND all_includes ${CMAKE_SOURCE_DIR}/src/deploy/${dep}/cpp)
    endforeach()
    list(APPEND all_includes ${extra_includes})

    target_include_directories(${target_name} PRIVATE ${all_includes})
    install(TARGETS ${target_name} DESTINATION bin)
endfunction()

# -----------------------------------------------------------------------------
# Applications (per-app cpp under applications/<name>/cpp/)
# (target_name cpp_file deploy_dir [deploy_dir ...] [EXTRA_INCLUDES path ...])
# -----------------------------------------------------------------------------
add_application(example_emotion emotion_detection/cpp/example_emotion.cpp yolov5_face emotion)
add_application(example_fall_detection fall_detection/cpp/example_fall_detection.cpp yolov8_pose)
add_application(example_intrusion_identification intrusion_detection/cpp/example_intrusion_identification.cpp bytetrack
    EXTRA_INCLUDES ${CMAKE_SOURCE_DIR}/../../rvv_libs/eigen-rvv/eigen)
add_application(example_fire_detection fire_detection/cpp/example_fire_detection.cpp yolov8)

