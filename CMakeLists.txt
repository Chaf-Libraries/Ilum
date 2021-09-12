cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(IlumEngine VERSION 0.0.1)
message(STATUS "[Project] ${PROJECT_NAME} ${PROJECT_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")
# list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/External/SDL2/cmake")

include(Init)
Init_Project()

add_definitions(-DENGINE_VERSION="0.0.1")

find_package(SDL2 QUIET)
if(NOT ${SDL2_FOUND})
    find_package(SDL2 REQUIRED PATHS ${PROJECT_SOURCE_DIR}/External/SDL2/cmake)
endif()

find_package(spdlog QUIET)
if(NOT ${spdlog_FOUND})
    find_package(spdlog REQUIRED PATHS ${PROJECT_SOURCE_DIR}/External/spdlog/lib/cmake/spdlog)
endif()

add_subdirectory(Source)
add_subdirectory(Test)
