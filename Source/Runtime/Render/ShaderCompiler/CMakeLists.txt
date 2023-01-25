set(TARGET_NAME ShaderCompiler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.inl")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

add_library(${TARGET_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

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
target_link_libraries(${TARGET_NAME} PRIVATE glslang::glslang)
 target_link_libraries(${TARGET_NAME} PRIVATE glslang::SPIRV)
 target_link_libraries(${TARGET_NAME} PRIVATE spirv_reflect)
 target_link_libraries(${TARGET_NAME} PRIVATE spirv-cross-hlsl)
 target_link_libraries(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/Source/External/slang/bin/windows-x64/release/slang.lib)
 target_link_libraries(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/Source/External/dxc/lib/x64/dxcompiler.lib)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Public")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Private")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Public/ShaderCompiler")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/External/slang")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/External/dxc/inc")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")

target_precompile_headers(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Public/ShaderCompiler/Precompile.hpp)

set_target_properties(ShaderCompiler PROPERTIES FOLDER "Runtime/Render")
