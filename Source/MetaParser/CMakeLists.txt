set(TARGET_NAME MetaParser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_DEBUG_POSTFIX )

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

target_link_libraries(${TARGET_NAME} PRIVATE Mustache)
target_link_libraries(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/Source/Dependencies/llvm/lib/x64/libclang.lib)

target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/Source/Dependencies/llvm/include)
target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/Source/Dependencies/Mustache)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")
target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX->")

target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")

target_precompile_headers(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Precompile.hpp)
set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_definitions(-DTIXML_USE_STL)
add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

add_custom_command(
	TARGET ${TARGET_NAME} POST_BUILD 
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/Source/Dependencies/llvm/bin/x64/libclang.dll"
        "${PROJECT_SOURCE_DIR}/bin"
)