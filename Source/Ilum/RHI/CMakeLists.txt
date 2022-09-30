set(TARGET_NAME RHI)

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
target_link_libraries(${TARGET_NAME} PRIVATE vulkan)
target_link_libraries(${TARGET_NAME} PRIVATE volk)
target_link_libraries(${TARGET_NAME} PRIVATE vma)
target_link_libraries(${TARGET_NAME} PRIVATE D3D12MemoryAllocator)
target_link_libraries(${TARGET_NAME} PRIVATE cuda.lib)
target_link_libraries(${TARGET_NAME} PRIVATE cudart.lib)
target_link_libraries(${TARGET_NAME} PRIVATE glad)
target_link_libraries(${TARGET_NAME} PRIVATE opengl32.lib)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/Ilum")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DRHI_BACKEND_VULKAN")


file(GLOB REFLECTION_FILES "*.hpp")

# add_custom_command(
# 	TARGET ${TARGET_NAME} POST_BUILD 
# 	COMMAND python ${PROJECT_SOURCE_DIR}/Scripts/Preprocess.py ${PROJECT_SOURCE_DIR}/Source/_Generate/${TARGET_NAME}_Generate.meta ${HEADER_FILES} ${SOURCE_FILES}
# )

target_precompile_headers(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Precompile.hpp)
set_target_properties(RHI PROPERTIES FOLDER Ilum)