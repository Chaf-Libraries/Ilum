message(STATUS "include Download.cmake successfully")

function(IsFileNotExist res filename hash_type hash)
    if(EXISTS ${filename})
        file(${hash_type} ${filename} originFileHash)
        string(TOLOWER ${hash} lhash)
        string(TOLOWER ${originFileHash} loriginFileHash)
        if(${lhash} STREQUAL ${loriginFileHash})
            set(${res} "FALSE" PARENT_SCOPE)
        endif()
    endif()
    set(${res} "TRUE" PARENT_SCOPE)
endfunction()

function(DownloadFile url filename hash_type hash)
    string(FIND "${url}" "." idx REVERSE)
    string(LENGTH "${url}" len)
    string(SUBSTRING "${url}" ${idx} ${len} suffix)
    string(REPLACE "." "" suffix ${suffix})
    if("${suffix}" STREQUAL "zip")
        set(filename ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/${filename})
    endif()
    IsFileNotExist(need ${filename} ${hash_type} ${hash})
    if(${need} STREQUAL "FALSE")
        message(STATUS "Found File: ${filename}")
        return()
    endif()
    string(REGEX MATCH ".*/" dir ${filename})
    file(MAKE_DIRECTORY ${dir})
    message(STATUS "Download File: ${filename}")
    file(DOWNLOAD ${url} ${filename}
        #TIMEOUT 120 # seconds
            SHOW_PROGRESS
        EXPECTED_HASH ${hash_type}=${hash}
        TLS_VERIFY ON
    )
    if("${suffix}" STREQUAL "zip")
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
endfunction()
