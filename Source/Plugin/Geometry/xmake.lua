function add_geometry_plugin(category, name)
    target(string.format("Geometry.%s.%s", category, name))

    set_kind("shared")
    set_group("Plugin/Geometry")

    add_rules("plugin")
    add_files(string.format("%s/%s.cpp", category, name))
    add_deps("Core", "Geometry")

    target("Plugin")
        add_deps(string.format("Geometry.%s.%s", category, name))
    target_end()

    target_end()
end

add_geometry_plugin("Subdivision", "Loop")