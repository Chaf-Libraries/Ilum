set_project("Ilum")
set_version("0.0.1")

set_xmakever("2.7.5")

set_warnings("all")
set_languages("c++17")

add_rules("mode.debug", "mode.release")

set_runtimes("MD")
add_defines("NOMINMAX")
set_warnings("all")

includes("xmake/module.lua")
includes("xmake/plugin.lua")
includes("xmake/shader.lua")
includes("Source")
