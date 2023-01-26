add_requires("cereal")
add_requires("glfw", {configs = {shared = true}})
add_requires("spdlog")
add_requires("stb")
add_requires("glm")
add_requires("glslang")
add_requires("spirv-cross")
add_requires("directxshadercompiler")
add_requires("assimp", {configs = {shared = true}})
add_requires("imgui docking")
add_requires("nativefiledialog")
add_requires("meshoptimizer")
add_requires("mustache")
add_requires("slang")
add_requires("volk", {configs = {header_only = true}})
add_requires("vulkan-headers")
add_requires("vulkan-memory-allocator")

option("CUDA_ENABLE")
    on_check(function (option)
        import("detect.sdks.find_cuda")
        local cuda = find_cuda()
        if cuda then
            option:enable(true)
            option:add("defines", "CUDA_ENABLE")
        else
            option:enable(false)
        end
    end)
option_end()

if has_config("CUDA_ENABLE") then
    add_requires("cuda")
end

target("ImGui-Tools")
    set_kind("static")

    add_files("imgui_tools/**.cpp")
    add_headerfiles("imgui_tools/**.h")
    add_includedirs("imgui_tools/", {public  = true})
    add_packages("glfw", "imgui")

    set_group("External")
target_end()

target("spirv-reflect")
    set_kind("static")

    add_files("spirv_reflect/**.c")
    add_headerfiles("spirv_reflect/**.h")
    add_includedirs("spirv_reflect", {public  = true})

    set_group("External")
target_end()

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
        result.links = "slang"
        result.includedirs = package:installdir()
        result.linkdirs = package:installdir("bin/windows-x64/release")
        return result
    end)
package_end()

