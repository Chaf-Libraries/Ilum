SetTarget(
    MODE STATIC
    TARGET_NAME Asset
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
    LIB
        stb
        meshoptimizer
        assimp::assimp
        Geometry
)
