add_runtime_moulde(
    "ShaderCompiler",
    "Runtime/Render",
    false, 
    {"Core", "RHI"}, 
    {"glslang", "spirv-cross", "spirv-reflect", "directxshadercompiler", "slang"}
)

add_runtime_moulde(
    "RenderGraph",
    "Runtime/Render",
    false, 
    {"Core", "RHI", "ShaderCompiler"}, 
    {}
)

add_runtime_moulde(
    "Material",
    "Runtime/Render",
    false, 
    {"Core", "RHI"}, 
    {}
)

add_runtime_moulde(
    "Renderer",
    "Runtime/Render",
    false, 
    {"Core", "RHI", "ShaderCompiler", "RenderGraph", "Scene", "Geometry", "Resource", "Material"}, 
    {}
)

-- Shader Compiler
-- target("ShaderCompiler")
--     if is_mode("debug") then
--         set_kind("shared")
--         add_rules("utils.symbols.export_all", {export_classes = true})
--     elseif is_mode("release") then
--         set_kind("static")
--     end

--     add_files("ShaderCompiler/Private/**.cpp")
--     add_headerfiles("ShaderCompiler/Public/**.hpp")

--     add_includedirs("ShaderCompiler/Public/ShaderCompiler")
--     add_includedirs("ShaderCompiler/Public", {public  = true})

--     add_deps("Core", "RHI")

--     add_packages("glslang", "spirv-cross", "spirv-reflect", "directxshadercompiler", "slang")

--     set_group("Runtime/Render")
-- target_end()

-- Render Graph
-- target("RenderGraph")
--     if is_mode("debug") then
--         add_defines("DEBUG")
--         set_kind("shared")
--         add_rules("utils.symbols.export_all", {export_classes = true})
--     elseif is_mode("release") then
--         set_kind("static")
--     end

--     add_files("RenderGraph/Private/**.cpp")
--     add_headerfiles("RenderGraph/Public/**.hpp")

--     add_includedirs("RenderGraph/Public/RenderGraph")
--     add_includedirs("RenderGraph/Public", {public  = true})

--     add_deps("Core", "RHI", "ShaderCompiler")

--     set_group("Runtime/Render")
-- target_end()

-- Material
-- target("Material")
--     if is_mode("debug") then
--         add_defines("DEBUG")
--         set_kind("shared")
--         add_rules("utils.symbols.export_all", {export_classes = true})
--     elseif is_mode("release") then
--         set_kind("static")
--     end

--     add_files("Material/Private/**.cpp")
--     add_headerfiles("Material/Public/**.hpp")

--     add_includedirs("Material/Public/Material")
--     add_includedirs("Material/Public", {public  = true})

--     add_deps("Core", "RHI")

--     set_group("Runtime/Render")
-- target_end()


-- Renderer
-- target("Renderer")
--     if is_mode("debug") then
--         add_defines("DEBUG")
--         set_kind("shared")
--         add_rules("utils.symbols.export_all", {export_classes = true})
--     elseif is_mode("release") then
--         set_kind("static")
--     end

--     add_files("Renderer/Private/**.cpp")
--     add_headerfiles("Renderer/Public/**.hpp")

--     add_includedirs("Renderer/Public/Renderer")
--     add_includedirs("Renderer/Public", {public  = true})

--     add_deps("Core", "RHI", "ShaderCompiler", "RenderGraph", "Scene",
--         "Geometry", "Resource", "Material")

--     set_group("Runtime/Render")
-- target_end()
