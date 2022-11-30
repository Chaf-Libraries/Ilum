set(TARGET_NAME External)

add_custom_target(${TARGET_NAME})

add_custom_command(
	TARGET ${TARGET_NAME} PRE_BUILD 
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/assimp/bin/assimp-vc143-mt.dll"
        "${PROJECT_SOURCE_DIR}/bin/assimp-vc143-mt.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CMAKE_CURRENT_SOURCE_DIR}/assimp/bin/assimp-vc143-mtd.dll"
		"${PROJECT_SOURCE_DIR}/bin/assimp-vc143-mtd.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CMAKE_CURRENT_SOURCE_DIR}/glfw/bin/glfw3.dll"
		"${PROJECT_SOURCE_DIR}/bin/glfw3.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CMAKE_CURRENT_SOURCE_DIR}/spdlog/bin/spdlog.dll"
		"${PROJECT_SOURCE_DIR}/bin/spdlog.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CMAKE_CURRENT_SOURCE_DIR}/spdlog/bin/spdlogd.dll"
		"${PROJECT_SOURCE_DIR}/bin/spdlogd.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/dxc/bin/x64/dxil.dll"
        "${PROJECT_SOURCE_DIR}/bin/dxil.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/dxc/bin/x64/dxcompiler.dll"
        "${PROJECT_SOURCE_DIR}/bin/dxcompiler.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/slang/bin/windows-x64/release/slang.dll"
        "${PROJECT_SOURCE_DIR}/bin/slang.dll"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/meshoptimizer/bin/meshoptimizer.dll"
        "${PROJECT_SOURCE_DIR}/bin/meshoptimizer.dll"
)

add_subdirectory(stb)
add_subdirectory(vulkan)
add_subdirectory(imgui)
add_subdirectory(spirv_reflect)
add_subdirectory(nfd)

set_target_properties(External PROPERTIES FOLDER External)
set_target_properties(stb PROPERTIES FOLDER External)
set_target_properties(vulkan PROPERTIES FOLDER External)
set_target_properties(imgui PROPERTIES FOLDER External)
set_target_properties(spirv_reflect PROPERTIES FOLDER External)
set_target_properties(nfd PROPERTIES FOLDER External)