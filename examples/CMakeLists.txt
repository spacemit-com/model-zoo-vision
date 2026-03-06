cmake_minimum_required(VERSION 3.10)

# Link libvision.so (shared; self-registration is included in the .so).
function(link_vision_with_registration target_name)
    target_link_libraries(${target_name} PRIVATE vision)
endfunction()

# Function to create an example executable (examples use only vision_service.h API)
function(add_model_example example_name deploy_dir)
    set(cpp_file "${CMAKE_CURRENT_SOURCE_DIR}/${example_name}/cpp/${example_name}.cpp")
    
    if(EXISTS ${cpp_file})
        add_executable(${example_name} ${example_name}/cpp/${example_name}.cpp)
        link_vision_with_registration(${example_name})
        target_link_libraries(${example_name} PRIVATE ${LIBS})
        
        # Include directories: public API; src for common/cpp (e.g. image_processing) when needed
        target_include_directories(${example_name} PRIVATE
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/src
        )
        
        # Add to install list
        set(EXAMPLE_TARGETS ${EXAMPLE_TARGETS} ${example_name} PARENT_SCOPE)
    endif()
endfunction()

# Initialize example targets list
set(EXAMPLE_TARGETS)

# Define all examples with their corresponding deploy directories
add_model_example(yolov8 yolov8)
add_model_example(yolov5 yolov5)
add_model_example(yolov11 yolov11)
add_model_example(yolov8_pose yolov8_pose)
add_model_example(yolov8_seg yolov8_seg)
add_model_example(pp_liteseg pp_liteseg)
add_model_example(arcface arcface)
add_model_example(yolov5-face yolov5_face)
add_model_example(yolov5_gesture yolov5_gesture)
add_model_example(resnet resnet)
add_model_example(emotion emotion)
add_model_example(bytetrack bytetrack)
add_model_example(ocsort ocsort)

# Install examples (only install targets that were actually created)
if(EXAMPLE_TARGETS)
    install(TARGETS ${EXAMPLE_TARGETS} DESTINATION bin)
endif()