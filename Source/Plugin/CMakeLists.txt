add_custom_target(Plugins)
set_target_properties(Plugins PROPERTIES FOLDER "Plugin/")

function(add_plugin name category source)
    add_library(${name} MODULE ${source})
    set_target_properties(${name} PROPERTIES FOLDER "Plugin/${category}")
    add_dependencies(plugins ${name})
endfunction()

add_subdirectory(RHI)