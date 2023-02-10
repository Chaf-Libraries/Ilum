target("Geometry")
    set_kind("$(kind)")
    add_files("Private/**.cpp")
    set_pcxxheader("Public/Geometry/Precompile.hpp")

    add_includedirs("Public/Geometry")
    add_includedirs("Public", {public  = true})

    add_deps("Core")
    add_packages("spdlog", "glm", "cereal")
target_end()
