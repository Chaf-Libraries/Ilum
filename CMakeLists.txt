cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(IlumEngine VERSION 0.0.1)
message(STATUS "[Project] ${PROJECT_NAME} ${PROJECT_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")

include(Init)
Init_Project()

add_definitions(-DENGINE_VERSION="0.0.1")

find_package(sdl2 REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/sdl2/cmake)
find_package(tbb REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/tbb/lib/cmake/TBB)
find_package(assimp REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/assimp/lib/cmake/assimp)
find_package(glslang REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/glslang/lib/cmake)
find_package(spirv_cross REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/spirv_cross/share)
find_package(volk REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/volk/lib/cmake/volk)
find_package(meshoptimizer REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/meshoptimizer/lib/cmake/meshoptimizer)
find_package(spdlog REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/Dependencies/spdlog/lib/cmake/spdlog)

add_subdirectory(Dependencies)

macro(GET_ALL_TARGETS TARGETS DIR)
    get_property(SUBDIRECTORIES DIRECTORY ${DIR} PROPERTY SUBDIRECTORIES)
    foreach(SUBDIR ${SUBDIRECTORIES})
        GET_ALL_TARGETS(${TARGETS} ${SUBDIR})
    endforeach()
    get_property(CUR_TARGETS DIRECTORY ${DIR} PROPERTY BUILDSYSTEM_TARGETS)
    LIST(APPEND ${TARGETS} ${CUR_TARGETS})
endmacro()

GET_ALL_TARGETS(ALL_TARGETS ${CMAKE_CURRENT_SOURCE_DIR})

FOREACH(T IN ITEMS ${ALL_TARGETS})
	SET_TARGET_PROPERTIES(${T} PROPERTIES FOLDER "External")
ENDFOREACH()

add_subdirectory(Source)
