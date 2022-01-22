SetTarget(
    MODE STATIC
    TARGET_NAME Core
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
        USE_GLFW
        USE_PMR
    INC
        ${PROJECT_SOURCE_DIR}/Source
    LIB
        spdlog
        glfw
        SDL2::SDL2
)