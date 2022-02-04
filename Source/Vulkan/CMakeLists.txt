SetTarget(
    MODE STATIC
    TARGET_NAME Vulkan
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
    LIB
        Core
        vulkan
        volk
        vma
        glslang
        SPIRV
        spirv-cross-glsl
)