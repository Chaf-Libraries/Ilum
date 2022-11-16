set(TARGET_NAME RHI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.hpp")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

file(GLOB_RECURSE PUBLIC_FILES "Public/*.hpp" "Public/*.cpp")
file(GLOB PRIVATE_FILES "Private/*.hpp" "Private/*.cpp")

file(GLOB_RECURSE VULKAN_FILES "Private/Backend/Vulkan/*.hpp" "Private/Backend/Vulkan/*.cpp")
file(GLOB_RECURSE DX12_FILES "Private/Backend/DX12/*.hpp" "Private/Backend/DX12/*.cpp")
file(GLOB_RECURSE CUDA_FILES "Private/Backend/CUDA/*.hpp" "Private/Backend/CUDA/*.cpp")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${PUBLIC_FILES} ${PRIVATE_FILES} ${VULKAN_FILES} ${DX12_FILES} ${CUDA_FILES})

if(ENABLE_CUDA)
	add_library(${TARGET_NAME} ${PUBLIC_FILES} ${PRIVATE_FILES} ${VULKAN_FILES} ${DX12_FILES} ${CUDA_FILES})
else()
	add_library(${TARGET_NAME} ${PUBLIC_FILES} ${PRIVATE_FILES} ${VULKAN_FILES} ${DX12_FILES})
endif()

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

file(GLOB REFLECTION_FILES "*.hpp")

set_target_properties(RHI PROPERTIES FOLDER Runtime)