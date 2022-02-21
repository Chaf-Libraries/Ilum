SetTarget(
    MODE STATIC
    TARGET_NAME Render
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/Source/Render
    LIB
        Graphics
        Resource
        imgui-node-editor
)