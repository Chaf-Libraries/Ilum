
        message(WARNING "Using `HLSLTargets.cmake` is deprecated: use `find_package(glslang)` to find glslang CMake targets.")

        if (NOT TARGET glslang::HLSL)
            include("${CMAKE_CURRENT_LIST_DIR}/../../lib/cmake/glslang/glslang-targets.cmake")
        endif()

        add_library(HLSL ALIAS glslang::HLSL)
    