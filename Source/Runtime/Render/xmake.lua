add_runtime_moulde(
    "ShaderCompiler",
    "Runtime/Render",
    false, 
    {"Core", "RHI"}, 
    {"glslang", "spirv-cross", "directxshadercompiler", "slang", "spirv-reflect"}
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
