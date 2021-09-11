message(STATUS "include Git.cmake successfully")

macro(Git_Init)
    message(STATUS "--- Git Init ---")
    find_package(Git REQUIRED)
    message(STATUS "GIT_FOUND: ${GIT_FOUND}")
    message(STATUS "GIT_EXECUTABLE: ${GIT_EXECUTABLE}")
    message(STATUS "GIT_VERSION_STRING: ${GIT_VERSION_STRING}")
endmacro()

function(SubModuleUpdate)
    if(NOT GIT_FOUND)
        Git_Init()
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule init
        #OUTPUT_VARIABLE out
        #OUTPUT_STRIP_TRAILING_WHITESPACE
        #ERROR_QUIET
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update
        #OUTPUT_VARIABLE out
        #OUTPUT_STRIP_TRAILING_WHITESPACE
        #ERROR_QUIET
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endfunction()
