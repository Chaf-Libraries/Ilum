function add_importer_plugin(name, deps, pkgs)
    target(string.format("Importer.%s", name))

    set_kind("shared")
    set_group("Plugin/Importer")

    add_rules("plugin")
    add_files(string.format("%s.cpp", name))
    add_deps("Core", "RHI", "Resource", deps)
    add_packages(pkgs)

    target("Plugin")
        add_deps(string.format("Importer.%s", name))
    target_end()

    target_end()
end

add_importer_plugin("STB", {}, {"stb"})
add_importer_plugin("DDS", {}, {})
add_importer_plugin("Assimp", {"Geometry"}, {"assimp", "stb", "meshoptimizer"})
add_importer_plugin("USD", {"Geometry"}, {"usd", "stb", "meshoptimizer"})