# SDL
option(VIDEO_OPENGLES "Include OpenGL ES support" OFF)
add_subdirectory(SDL)

# spdlog
add_subdirectory(spdlog)

#vulkan
add_library(vulkan INTERFACE)
set(VULKAN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/include)
target_sources(vulkan INTERFACE ${VULKAN_INCLUDE_DIR}/vulkan/vulkan.h)
target_include_directories(vulkan INTERFACE ${VULKAN_INCLUDE_DIR})
target_compile_definitions(vulkan INTERFACE VK_NO_PROTOTYPES)

if(ANDROID)
    target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_ANDROID_KHR)
elseif(WIN32)
    target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_WIN32_KHR)
elseif(APPLE)
    target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_MACOS_MVK)
elseif(UNIX)
    # See whether X11 is available. If not, fall back to direct-to-display mode.
    find_package(X11 QUIET)
    if (X11_FOUND)
        target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_XCB_KHR)
    else()
        set(DIRECT_TO_DISPLAY TRUE)
        set(DIRECT_TO_DISPLAY TRUE PARENT_SCOPE)
        target_compile_definitions(vulkan INTERFACE VK_USE_PLATFORM_DISPLAY_KHR)
    endif()
endif() 

# vma
add_library(vma INTERFACE)
set(VMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vma/include)
target_sources(vma INTERFACE ${VMA_DIR}/vk_mem_alloc.h)
target_include_directories(vma INTERFACE ${VMA_DIR})
target_link_libraries(vma INTERFACE vulkan)

# volk
set(VOLK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/volk")
set(VOLK_FILES
    "${VOLK_DIR}/volk.c"
    "${VOLK_DIR}/volk.h")
add_library(volk STATIC ${VOLK_FILES})
target_link_libraries(volk PUBLIC vulkan)
target_include_directories(volk PUBLIC ${VOLK_DIR})
# set_property(TARGET volk PROPERTY FOLDER "Extern/volk")

# imgui
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
set(IMGUI_FILES
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imconfig.h"
    "${IMGUI_DIR}/imgui.h"
    "${IMGUI_DIR}/imgui_internal.h"
    "${IMGUI_DIR}/imstb_rectpack.h"
    "${IMGUI_DIR}/imstb_textedit.h"
    "${IMGUI_DIR}/imstb_truetype.h"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.h"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_sdl.h"
    "${IMGUI_DIR}/backends/imgui_impl_sdl.cpp")
add_library(imgui STATIC ${IMGUI_FILES})
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui SDL2 volk)

# glslang
option(ENABLE_SPVREMAPPER OFF)
option(ENABLE_GLSLANG_BINARIES OFF)
option(ENABLE_HLSL ON)
option(ENABLE_OPT OFF)
option(BUILD_TESTING OFF)
option(BUILD_EXTERNAL OFF)

add_subdirectory(glslang)

if (NOT TARGET glslang-default-resource-limits)
    add_library(glslang-default-resource-limits
                glslang/StandAlone/ResourceLimits.cpp)

    # set_property(TARGET glslang-default-resource-limits PROPERTY FOLDER "Extern/glslang")
    
    target_include_directories(glslang-default-resource-limits
                                PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glslang/StandAlone)
endif()

if(NOT MSVC)
    target_compile_options(glslang PRIVATE
        "-Wno-logical-op-parentheses"
        "-Wno-unused-parameter")

    target_compile_options(SPIRV PRIVATE
        "-Wno-logical-op-parentheses"
        "-Wno-unused-parameter")
endif()

# spirv-cross
add_subdirectory(spirv_cross)

#assimp
option(ASSIMP_BUILD_TESTS OFF)
add_subdirectory(assimp)

#entt
add_library(entt INTERFACE)
target_include_directories(entt INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/entt/single_include/entt")

#stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/stb")

# yaml-cpp
add_subdirectory(yaml-cpp)

# glm
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm")

# ImGuizmo
set(IMGUIZMO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo")
set(IMGUIZMO_FILES
    "${IMGUIZMO_DIR}/GraphEditor.cpp"
    "${IMGUIZMO_DIR}/GraphEditor.h"
    "${IMGUIZMO_DIR}/ImCurveEdit.cpp"
    "${IMGUIZMO_DIR}/ImCurveEdit.h"
    "${IMGUIZMO_DIR}/ImGradient.cpp"
    "${IMGUIZMO_DIR}/ImGradient.h"
    "${IMGUIZMO_DIR}/ImGuizmo.cpp"
    "${IMGUIZMO_DIR}/ImGuizmo.h"
    "${IMGUIZMO_DIR}/ImSequencer.cpp"
    "${IMGUIZMO_DIR}/ImSequencer.h"
    "${IMGUIZMO_DIR}/ImZoomSlider.h")
add_library(ImGuizmo STATIC ${IMGUIZMO_FILES})
target_include_directories(ImGuizmo PUBLIC ${IMGUIZMO_DIR})
target_link_libraries(ImGuizmo imgui)

# imgui-node-editor
set(IMGUINODEEDITOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui_node_editor")
set(IMGUINODEEDITOR_FILES
    "${IMGUINODEEDITOR_DIR}/crude_json.cpp"
    "${IMGUINODEEDITOR_DIR}/crude_json.h"
    "${IMGUINODEEDITOR_DIR}/imgui_bezier_math.inl"
    "${IMGUINODEEDITOR_DIR}/imgui_bezier_math.h"
    "${IMGUINODEEDITOR_DIR}/imgui_canvas.cpp"
    "${IMGUINODEEDITOR_DIR}/imgui_canvas.h"
    "${IMGUINODEEDITOR_DIR}/imgui_extra_math.h"
    "${IMGUINODEEDITOR_DIR}/imgui_extra_math.inl"
    "${IMGUINODEEDITOR_DIR}/imgui_node_editor.cpp"
    "${IMGUINODEEDITOR_DIR}/imgui_node_editor.h"
    "${IMGUINODEEDITOR_DIR}/imgui_node_editor_api.cpp"
    "${IMGUINODEEDITOR_DIR}/imgui_node_editor_internal.h"
    "${IMGUINODEEDITOR_DIR}/imgui_node_editor_internal.inl"
    "${IMGUINODEEDITOR_DIR}/examples/blueprints-example/utilities/builders.h"
    "${IMGUINODEEDITOR_DIR}/examples/blueprints-example/utilities/builders.cpp"
    "${IMGUINODEEDITOR_DIR}/examples/blueprints-example/utilities/drawing.h"
    "${IMGUINODEEDITOR_DIR}/examples/blueprints-example/utilities/drawing.cpp"
    "${IMGUINODEEDITOR_DIR}/examples/blueprints-example/utilities/widgets.h"
    "${IMGUINODEEDITOR_DIR}/examples/blueprints-example/utilities/widgets.cpp")
add_library(imgui-node-editor STATIC ${IMGUINODEEDITOR_FILES})
target_include_directories(imgui-node-editor PUBLIC ${IMGUINODEEDITOR_DIR} ${IMGUINODEEDITOR_DIR}/examples/blueprints-example/utilities)
target_link_libraries(imgui-node-editor imgui)

# ImFileDialog
set(IMFILEDIALOG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ImFileDialog")
set(IMFILEDIALOG_FILES
    "${IMFILEDIALOG_DIR}/ImFileDialog.cpp"
    "${IMFILEDIALOG_DIR}/ImFileDialog.h")
add_library(ImFileDialog STATIC ${IMFILEDIALOG_FILES})
target_include_directories(ImFileDialog PUBLIC ${IMFILEDIALOG_DIR})
target_include_directories(ImFileDialog PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ImFileDialog imgui stb)

# mesh optimizer
add_subdirectory(meshoptimizer)