SetTarget(
    MODE STATIC
    TARGET_NAME ImGui
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/Source/ImGui
    LIB
        Graphics
        Resource
        imnodes
)