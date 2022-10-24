file(GLOB_RECURSE SOURCE_FILES "*.cpp")

pybind11_add_module(Ilum ${SOURCE_FILES})

target_link_libraries(Ilum PRIVATE Core)
target_link_libraries(Ilum PRIVATE RHI)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${SOURCE_FILES})