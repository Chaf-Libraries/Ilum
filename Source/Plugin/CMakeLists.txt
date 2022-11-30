add_custom_target(Plugins)
set_target_properties(Plugins PROPERTIES FOLDER "Plugin/")

add_subdirectory(RHI)
add_subdirectory(Pass)
add_subdirectory(Editor)
add_subdirectory(Importer)