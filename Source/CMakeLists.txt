SetTarget(
    MODE STATIC
    TARGET_NAME Ilum
    DEFINE
        PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}/"
        NOMINMAX
    INC
        ${PROJECT_SOURCE_DIR}/Source
        ${PROJECT_SOURCE_DIR}/External/SDL2/include
        ${PROJECT_SOURCE_DIR}/External/spdlog/include
        ${PROJECT_SOURCE_DIR}/External/volk/include
        ${PROJECT_SOURCE_DIR}/External/vma/include
        ${Vulkan_INCLUDE_DIRS}
    LIB
        SDL2::SDL2
        spdlog::spdlog
        volk::volk
        glslang
        SPIRV
        spirv-cross-glsl
        Vulkan::Vulkan
        stb
)