set(TARGET_NAME Engine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.hpp" "*.h")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.c")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

add_executable(${TARGET_NAME} ${HEADER_FILES} ${SOURCE_FILES})

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
target_link_libraries(${TARGET_NAME} PRIVATE RenderCore)
# target_link_libraries(${TARGET_NAME} PRIVATE spdlog)
# target_link_libraries(${TARGET_NAME} PRIVATE glm)
# target_link_libraries(${TARGET_NAME} PRIVATE vulkan)
# target_link_libraries(${TARGET_NAME} PRIVATE volk)
# target_link_libraries(${TARGET_NAME} PRIVATE vma)
# target_link_libraries(${TARGET_NAME} PRIVATE stb)
# target_link_libraries(${TARGET_NAME} PRIVATE glslang)
# target_link_libraries(${TARGET_NAME} PRIVATE SPIRV)
# target_link_libraries(${TARGET_NAME} PRIVATE dxc)
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

# target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/Ilum")
# target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/Dependencies/dxc/inc")
# target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/Dependencies/FontAwesome")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")
# target_compile_definitions(${TARGET_NAME} PRIVATE "-DPROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
# target_precompile_headers(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Precompile.hpp)
set_target_properties(Engine PROPERTIES FOLDER Ilum)