set(TARGET_NAME RHI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

file(GLOB_RECURSE PUBLIC_FILES "Public/*.hpp" "Public/*.cpp")
file(GLOB PRIVATE_FILES "Private/*.hpp" "Private/*.cpp")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${PUBLIC_FILES} ${PRIVATE_FILES})

add_library(${TARGET_NAME} ${PUBLIC_FILES} ${PRIVATE_FILES})

set_target_properties(
	${TARGET_NAME} 
	PROPERTIES 
	CXX_STANDARD 17 
	DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
)

target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")
target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX->")

target_link_libraries(${TARGET_NAME} PRIVATE Core)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Public")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Private")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Public/RHI")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DRHI_BACKEND_VULKAN")

target_precompile_headers(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Public/RHI/Fwd.hpp)

set_target_properties(RHI PROPERTIES FOLDER Runtime)