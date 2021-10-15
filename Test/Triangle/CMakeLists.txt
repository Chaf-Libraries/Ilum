SetTarget(
    MODE EXE
    TARGET_NAME Triangle
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
    LIB
        Ilum
)