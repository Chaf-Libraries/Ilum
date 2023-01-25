set(TARGET_NAME Core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

add_library(${TARGET_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties(
	${TARGET_NAME} 
	PROPERTIES 
	CXX_STANDARD 17 
	DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
)

#set_property(${TARGET_NAME} APPEND PROPERTY LINK_FLAGS )

target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")
target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX->")

target_link_libraries(${TARGET_NAME} PRIVATE glfw)
target_link_libraries(${TARGET_NAME} PUBLIC spdlog::spdlog)
target_link_libraries(${TARGET_NAME} PUBLIC glm::glm)
target_link_libraries(${TARGET_NAME} PUBLIC cereal::cereal)
target_link_libraries(${TARGET_NAME} PUBLIC stb)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Public")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Private")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Public/Core")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/Dependencies/stb")

target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")

add_dependencies(${TARGET_NAME} External)

target_precompile_headers(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Public/Core/Precompile.hpp)

set_target_properties(Core PROPERTIES FOLDER Runtime)