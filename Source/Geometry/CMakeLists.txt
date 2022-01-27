SetTarget(
    MODE STATIC
    TARGET_NAME Geometry
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
    LIB
        glm
)
