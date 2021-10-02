cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(IlumEngine VERSION 0.0.1)
message(STATUS "[Project] ${PROJECT_NAME} ${PROJECT_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")
# list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/External/SDL2/cmake")

include(Init)
Init_Project()

add_definitions(-DENGINE_VERSION="0.0.1")

# SDL2
find_package(SDL2 QUIET)
if(NOT ${SDL2_FOUND})
    find_package(SDL2 REQUIRED PATHS ${PROJECT_SOURCE_DIR}/External/sdl2/cmake)
endif()

# spdlog
find_package(spdlog QUIET)
if(NOT ${spdlog_FOUND})
    find_package(spdlog REQUIRED PATHS ${PROJECT_SOURCE_DIR}/External/spdlog/lib/cmake/spdlog)
endif()

# volk
find_package(volk QUIET)
if(NOT ${volk_FOUND})
    find_package(volk REQUIRED PATHS ${PROJECT_SOURCE_DIR}/External/volk/lib/cmake/volk)
endif()

# glslang
find_package(glslang QUIET)
if(NOT ${glslang_FOUND})
    find_package(glslang REQUIRED PATHS ${PROJECT_SOURCE_DIR}/External/glslang/lib/cmake)
endif()

# spirv-cross
find_package(spirv_cross_core QUIET)
if(NOT ${spirv_cross_core_FOUND})
    find_package(spirv_cross_core REQUIRED PATHS ${PROJECT_SOURCE_DIR}/External/spirv_cross/share/spirv_cross_core/cmake)
endif()

find_package(spirv_cross_glsl QUIET)
if(NOT ${spirv_cross_glsl_FOUND})
    find_package(spirv_cross_glsl REQUIRED PATHS ${PROJECT_SOURCE_DIR}/External/spirv_cross/share/spirv_cross_glsl/cmake)
endif()

# vulkan
find_package(Vulkan REQUIRED)

# stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/External/stb")

add_subdirectory(Source)
add_subdirectory(Test)
