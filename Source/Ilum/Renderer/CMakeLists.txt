set(TARGET_NAME Renderer)

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
target_link_libraries(${TARGET_NAME} PRIVATE RenderCore)
target_link_libraries(${TARGET_NAME} PRIVATE Scene)
target_link_libraries(${TARGET_NAME} PRIVATE Resource)
target_link_libraries(${TARGET_NAME} PRIVATE rttr_core_lib)
target_link_libraries(${TARGET_NAME} PRIVATE entt)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/Ilum")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")

add_custom_command(
	TARGET ${TARGET_NAME} POST_BUILD 
	COMMAND ${PROJECT_SOURCE_DIR}/bin/CodeGeneration.exe ${PROJECT_SOURCE_DIR}/Source/CodeGeneration/Meta/RendererMeta.hpp ${HEADER_FILES}
)

target_precompile_headers(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Precompile.hpp)
set_target_properties(Renderer PROPERTIES FOLDER Ilum)