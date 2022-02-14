SetTarget(
    MODE STATIC
    TARGET_NAME Graphics
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
    LIB
        SDL2::SDL2
        Core
        vulkan
        volk
        vma
        glslang
        SPIRV
        spirv-cross-glsl
        imgui
)