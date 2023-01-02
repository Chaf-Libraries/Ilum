set(TARGET_NAME RHI.Vulkan)

file(GLOB HEADER_FILES "*.hpp" "*.h" "*.inl")
file(GLOB SOURCE_FILES "*.cpp" "*.c")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

add_library(${TARGET_NAME} MODULE ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Plugin/RHI")
add_dependencies(Plugins ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE Core)
target_link_libraries(${TARGET_NAME} PRIVATE RHI)
target_link_libraries(${TARGET_NAME} PRIVATE vulkan)

target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")

target_precompile_headers(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Fwd.hpp)
copy_shared_lib(${TARGET_NAME} RHI)
