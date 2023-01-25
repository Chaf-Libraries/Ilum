add_runtime_moulde(
    "Core",
    "Runtime",
    true, 
    {}, 
    {}
)

target("Core")
    add_packages("glfw", "spdlog", "glm", "cereal", "stb", {public = true})
target_end()

add_runtime_moulde(
    "RHI", 
    "Runtime",
    false, 
    {"Core"}, 
    {}
)

add_runtime_moulde(
    "Geometry", 
    "Runtime",
    false, 
    {"Core"}, 
    {"glm"}
)

add_runtime_moulde(
    "Scene", 
    "Runtime",
    false, 
    {"Core", "ImGui-Tools"},
    {"imgui"}
)

add_runtime_moulde(
    "Resource", 
    "Runtime",
    false, 
    {"Core", "RHI", "Geometry", "Material", "RenderGraph", "ShaderCompiler", "Scene"}, 
    {"imgui", "meshoptimizer", "mustache"}
)

-- Render
includes("Render")