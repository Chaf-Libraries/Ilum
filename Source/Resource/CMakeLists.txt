SetTarget(
    MODE STATIC
    TARGET_NAME Resource
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/Source/Resource
    LIB
        Graphics
        assimp::assimp
        stb
        glm
        meshoptimizer
)