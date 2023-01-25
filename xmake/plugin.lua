includes("rule.lua")

target("Plugin")
    add_rules("empty")
    set_group("Plugin")
target_end()

rule("plugin")
    after_build(function (target)
        local source_path = path.join("$(projectdir)", target:targetdir(), string.format("%s.dll", target:name()))
        local target_path = path.join("$(projectdir)", "shared", string.sub(target:name(), 0, string.find(target:name(), "%.") - 1), string.format("%s.dll", target:name()))
        os.cp(source_path, target_path)
    end)
