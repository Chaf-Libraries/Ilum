includes("External")
includes("Runtime")
includes("Plugin")

target("Editor")
    if is_mode("debug") then
        add_defines("DEBUG")
        set_kind("shared")
        add_rules("utils.symbols.export_all", {export_classes = true})
    elseif is_mode("release") then
        set_kind("static")
    end

    set_pcxxheader("Editor/Precompile.hpp")

    add_files("Editor/**.cpp")
    add_headerfiles("Editor/**.hpp")
    add_includedirs("./", {public  = true})
    add_deps("Core", "RHI", "Scene", "ShaderCompiler", "ImGui-Tools")
    add_packages("imgui", "imnodes", "nativefiledialog")
target_end()

target("Shader")
    add_rules("shader", "empty")
    add_headerfiles("Shaders/**.hlsl")
    add_headerfiles("Shaders/**.hlsli")
target_end()

target("Engine")
    set_kind("binary")
    set_default(true)
    set_rundir("$(projectdir)")
    set_pcxxheader("Engine/Precompile.hpp")

    add_files("Engine/**.cpp")
    add_headerfiles("Engine/**.hpp")
    add_includedirs("Engine", {public  = true})
    add_deps("Core", "RHI", "Scene", "Geometry", "Resource", "Renderer", "Editor", "Plugin")
target_end()