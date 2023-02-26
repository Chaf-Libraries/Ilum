target("RHI.Vulkan")
    set_kind("shared")

    add_options("CUDA_ENABLE")
    if has_config("CUDA_ENABLE") then
        add_links("advapi32")
    end

    add_rules("plugin")
    add_defines("VK_NO_PROTOTYPES")

    if is_plat("windows") then
        add_defines("VK_USE_PLATFORM_WIN32_KHR")
    end

    add_files("Vulkan/**.cpp")
    add_headerfiles("Vulkan/**.hpp")

    add_deps("Core", "RHI")
    add_packages("volk")

    target("Plugin")
        add_deps("RHI.Vulkan")
    target_end()

    set_group("Plugin/RHI")
target_end()

if has_config("CUDA_ENABLE") and is_plat("windows")  then
    target("RHI.CUDA")
        set_kind("shared")

        add_rules("plugin")

        add_files("CUDA/**.cpp")
        add_headerfiles("CUDA/**.hpp")

        add_deps("Core", "RHI")
        add_packages("volk", "cuda")
        add_links("cudart", "cuda")
        
        target("Plugin")
            add_deps("RHI.CUDA")
        target_end()

        set_group("Plugin/RHI")
    target_end()
end
