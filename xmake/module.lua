function add_runtime_moulde(name, group, pcxxheader, deps, pkgs)
    target(name)
        if is_mode("debug") then
            add_defines("DEBUG")
            set_kind("shared")
            add_rules("utils.symbols.export_all", {export_classes = true})
        elseif is_mode("release") then
            set_kind("static")
        end

        if pcxxheader then
            set_pcxxheader(string.format("%s/Public/%s/Precompile.hpp", name, name))
        end

        add_files(string.format("%s/Private/**.cpp", name))
        add_headerfiles(string.format("%s/Public/**.hpp", name))
        add_includedirs(string.format("%s/Public/%s", name, name))
        add_includedirs(string.format("%s/Public", name), {public  = true})
        add_deps(deps)
        add_packages(pkgs)

        set_group(group)
    target_end()
end