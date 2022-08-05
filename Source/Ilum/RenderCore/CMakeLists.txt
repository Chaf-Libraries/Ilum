set(TARGET_NAME RenderCore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

add_library(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties(
	${TARGET_NAME} 
	PROPERTIES 
	CXX_STANDARD 17 
	DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
)

target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")
target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX->")

target_link_libraries(${TARGET_NAME} PRIVATE Core)
target_link_libraries(${TARGET_NAME} PRIVATE RHI)

# target_link_libraries(${TARGET_NAME} PRIVATE glfw)
# target_link_libraries(${TARGET_NAME} PUBLIC spdlog)
# target_link_libraries(${TARGET_NAME} PUBLIC glm)
# target_link_libraries(${TARGET_NAME} PUBLIC stb)
target_link_libraries(${TARGET_NAME} PRIVATE glslang)

# target_link_libraries(${TARGET_NAME} PRIVATE stb)
# target_link_libraries(${TARGET_NAME} PRIVATE glslang)
 target_link_libraries(${TARGET_NAME} PRIVATE SPIRV)
 target_link_libraries(${TARGET_NAME} PRIVATE dxc)
 target_link_libraries(${TARGET_NAME} PRIVATE spirv-reflect-static)
# target_link_libraries(${TARGET_NAME} PRIVATE spirv-cross-reflect)
# target_link_libraries(${TARGET_NAME} PRIVATE spirv-cross-hlsl)
# target_link_libraries(${TARGET_NAME} PRIVATE spirv-cross-glsl)
# target_link_libraries(${TARGET_NAME} PRIVATE imgui)
# target_link_libraries(${TARGET_NAME} PRIVATE imnode)
# target_link_libraries(${TARGET_NAME} PRIVATE rttr_core_lib)
# target_link_libraries(${TARGET_NAME} PRIVATE cereal)
# target_link_libraries(${TARGET_NAME} PRIVATE ImFileDialog)
# target_link_libraries(${TARGET_NAME} PRIVATE entt)
# target_link_libraries(${TARGET_NAME} PRIVATE meshoptimizer)
# target_link_libraries(${TARGET_NAME} PRIVATE ImGuizmo)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/Ilum")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")
add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/Source/Dependencies/dxc/bin/x64/dxil.dll"
        "${PROJECT_SOURCE_DIR}/bin")

set_target_properties(RenderCore PROPERTIES FOLDER Ilum)