SetTarget(
    MODE EXE
    TARGET_NAME App
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
    LIB
        Ilum
)

add_custom_command(TARGET App POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${PROJECT_SOURCE_DIR}/Dependencies/TBB/bin"
        "${PROJECT_SOURCE_DIR}/Dependencies/assimp/bin"
        "${PROJECT_SOURCE_DIR}/Dependencies/sdl2/bin"
        $<TARGET_FILE_DIR:App>)