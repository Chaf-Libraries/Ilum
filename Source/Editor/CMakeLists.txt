set(TARGET_NAME Editor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

add_library(${TARGET_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties(
	${TARGET_NAME} 
	PROPERTIES 
	CXX_STANDARD 17 
	DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
	LINK_FLAGS "/NODEFAULTLIB:MSVCRT"
)

target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")
target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX->")

target_link_libraries(${TARGET_NAME} PRIVATE Core)
target_link_libraries(${TARGET_NAME} PRIVATE RHI)
target_link_libraries(${TARGET_NAME} PRIVATE SceneGraph)
target_link_libraries(${TARGET_NAME} PRIVATE ShaderCompiler)
target_link_libraries(${TARGET_NAME} PRIVATE imgui)
target_link_libraries(${TARGET_NAME} PRIVATE glfw)
target_link_libraries(${TARGET_NAME} PRIVATE nfd)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/Dependencies/FontAwesome")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source/Ilum")
target_include_directories(${TARGET_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/Source")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")
