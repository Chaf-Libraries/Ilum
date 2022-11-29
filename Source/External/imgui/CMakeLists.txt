set(TARGET_NAME imgui)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE HEADER_FILES "*.h")
file(GLOB_RECURSE SOURCE_FILES "*.cpp")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${HEADER_FILES} ${SOURCE_FILES})

add_library(${TARGET_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(imgui PRIVATE glfw)

target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")
target_compile_options(${TARGET_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX->")

# set_target_properties(RenderGraph PROPERTIES FOLDER "Runtime/Render")

# add_library(imgui STATIC
#     imconfig.h
#     imgui.cpp
#     imgui.h
#     imgui_demo.cpp
#     imgui_draw.cpp
#     imgui_impl_glfw.cpp
#     imgui_impl_glfw.h
#     imgui_internal.h
#     imgui_tables.cpp
#     imgui_widgets.cpp
#     imstb_rectpack.h
#     imstb_textedit.h
#     imstb_truetype.h
# )

# target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# target_link_libraries(imgui PRIVATE glfw)
