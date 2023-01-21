set(TARGET_NAME Resource)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")
file(GLOB_RECURSE INLINE_FILES "*.inl")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES} ${INLINE_FILES})

add_library(${TARGET_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${INLINE_FILES})

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
target_link_libraries(${TARGET_NAME} PRIVATE Geometry)
target_link_libraries(${TARGET_NAME} PRIVATE Material)
target_link_libraries(${TARGET_NAME} PRIVATE RenderGraph)
target_link_libraries(${TARGET_NAME} PRIVATE ShaderCompiler)
target_link_libraries(${TARGET_NAME} PRIVATE Scene)
target_link_libraries(${TARGET_NAME} PRIVATE meshoptimizer::meshoptimizer)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Public")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Private")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Public/Resource")
target_include_directories(${TARGET_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/Source/External/mustache")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")

set_target_properties(Resource PROPERTIES FOLDER Runtime)