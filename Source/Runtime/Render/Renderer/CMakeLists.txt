set(TARGET_NAME Renderer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.hpp" "*.cuh")
file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.cu")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

add_library(${TARGET_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

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
target_link_libraries(${TARGET_NAME} PRIVATE ShaderCompiler)
target_link_libraries(${TARGET_NAME} PRIVATE RenderGraph)
target_link_libraries(${TARGET_NAME} PRIVATE SceneGraph)
target_link_libraries(${TARGET_NAME} PRIVATE Components)
target_link_libraries(${TARGET_NAME} PRIVATE Geometry)
target_link_libraries(${TARGET_NAME} PRIVATE Resource)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Public")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Private")
target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Public/Renderer")
target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")

set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY ON)

set_target_properties(Renderer PROPERTIES FOLDER Runtime/Render)