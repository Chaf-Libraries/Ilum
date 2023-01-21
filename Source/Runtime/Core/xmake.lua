target("Core")
    set_kind("$(kind)")
    add_files("Private/*.cpp")
    set_pcxxheader("Public/Core/Precompile.hpp")

    add_includedirs("Public/Core")
    add_includedirs("Public", {public  = true})

    add_packages("glfw", "spdlog", "glm", "cereal", "stb")
target_end()
