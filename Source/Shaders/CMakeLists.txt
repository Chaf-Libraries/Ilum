file(GLOB_RECURSE inc_hlsl ${PROJECT_SOURCE_DIR}/Source/Shaders/*.hlsli)
file(GLOB_RECURSE src_hlsl ${PROJECT_SOURCE_DIR}/Source/Shaders/*.hlsl)
file(GLOB_RECURSE inc ${PROJECT_SOURCE_DIR}/Source/Shaders/*.hpp)

source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${inc_hlsl})
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${src_hlsl})
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${inc})
set_source_files_properties(${src_hlsl} PROPERTIES VS_TOOL_OVERRIDE None)

add_custom_target(Shaders
	ALL SOURCES ${inc_hlsl} ${src_hlsl} ${inc}
	DEPENDS ${inc_hlsl} ${src_hlsl} ${inc})
