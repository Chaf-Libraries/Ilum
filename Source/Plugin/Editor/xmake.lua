function add_editor_plugin(name, deps, pkgs)
    target(string.format("Editor.%s", name))

    set_kind("shared")
    set_group("Plugin/Editor")

    add_rules("plugin")
    add_files(string.format("%s.cpp", name))
    add_deps("Core", "RHI", "Renderer", "Editor", deps)
    add_packages("imgui", pkgs)

    target("Plugin")
        add_deps(string.format("Editor.%s", name))
    target_end()

    target_end()
end

add_editor_plugin("SceneView", {"Resource", "Scene", "RenderGraph"}, {})
add_editor_plugin("Hierarchy", {"Scene"}, {})
add_editor_plugin("Inspector", {"Scene", "RenderGraph"}, {})
add_editor_plugin("MainMenu", {"Scene"}, {"nativefiledialog"})
add_editor_plugin("MeshEditor", {"Resource", "Geometry"}, {})
add_editor_plugin("AnimationEditor", {"Resource", "Geometry"}, {"nativefiledialog"})
add_editor_plugin("ResourceBrowser", {"Resource"}, {"nativefiledialog"})
add_editor_plugin("RenderGraphEditor", {"Resource", "RenderGraph"}, {"nativefiledialog"})
add_editor_plugin("MaterialGraphEditor", {"Resource", "Material", "Geometry", "RenderGraph"}, {"nativefiledialog"})
