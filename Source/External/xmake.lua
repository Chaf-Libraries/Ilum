add_requires("assimp")
add_requires("cereal")
-- add_requires("directxshadercompiler")
add_requires("glfw")
add_requires("glm")
-- add_requires("glslang")
-- add_requires("imgui")
-- add_requires("nativefiledialog")
add_requires("spdlog")
-- add_requires("spirv-cross")
-- add_requires("spirv-reflect")
add_requires("stb")
-- add_requires("volk")
-- add_requires("vulkan-memory-allocator")

add_requires("meshoptimizer")
add_requires("mustache")
add_requires("slang")

package("meshoptimizer")
    on_load(function (package)
        package:set("installdir", path.join(os.scriptdir(), "meshoptimizer"))
    end)

    on_fetch(function (package)
        package:addenv("PATH", package:installdir("bin"))

        local result = {}
        result.links = "meshoptimizer"
        result.linkdirs = package:installdir("lib")
        result.includedirs = package:installdir("include")
        return result
    end)
package_end()

package("mustache")
    on_load(function (package)
        package:set("installdir", path.join(os.scriptdir(), "mustache"))
    end)

    on_fetch(function (package)
        local result = {}
        result.includedirs = package:installdir()
        return result
    end)
package_end()

package("slang")
    on_load(function (package)
        package:set("installdir", path.join(os.scriptdir(), "slang"))
    end)

    on_fetch(function (package)
        package:addenv("PATH", package:installdir("bin/windows-x64/release"))

        local result = {}
        result.includedirs = package:installdir()
        return result
    end)
package_end()
