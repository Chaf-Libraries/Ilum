function add_pass_plugin(category, name)
    target(string.format("RenderPass.%s.%s", category, name))

    set_kind("shared")
    set_group(string.format("Plugin/RenderPass/%s", category))

    add_rules("plugin")
    add_files(string.format("%s.cpp", name))
    add_headerfiles("IPass.hpp", "PassData.hpp")
    add_includedirs("./")
    add_deps("Core", "RHI", "RenderGraph", "Renderer", "Scene", "Resource", "Geometry", "Material")
    add_packages("imgui")

    target("Plugin")
        add_deps(string.format("RenderPass.%s.%s", category, name))
    target_end()

    target_end()
end

add_pass_plugin("Output", "Present")
add_pass_plugin("Transfer", "CopyImageRGBA16")
add_pass_plugin("RayTracing", "PathTracing")
add_pass_plugin("RayTracing", "RayTracedAO")
add_pass_plugin("RenderPath", "VisibilityGeometryPass")
add_pass_plugin("RenderPath", "VisibilityLightingPass")
add_pass_plugin("RenderPath", "VisibilityBufferVisualization")
add_pass_plugin("Shading", "IBL")
add_pass_plugin("Shading", "SkyBoxPass")
add_pass_plugin("Shading", "ShadowMapPass")
add_pass_plugin("Shading", "CompositePass")
add_pass_plugin("PostProcess", "Bloom")
add_pass_plugin("PostProcess", "FXAA")
add_pass_plugin("PostProcess", "Tonemapping")