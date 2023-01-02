add_custom_target(Plugins)
set_target_properties(Plugins PROPERTIES FOLDER "Plugin/")

function(copy_shared_lib target folder)
    add_custom_command(
        TARGET ${target} POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PROJECT_SOURCE_DIR}/lib/${target}.dll"
            "${PROJECT_SOURCE_DIR}/shared/${folder}/${target}.dll"
    )
endfunction()

add_subdirectory(RHI)
add_subdirectory(RenderPass)
add_subdirectory(MaterialNode)
add_subdirectory(Editor)
add_subdirectory(Importer)
add_subdirectory(Geometry)