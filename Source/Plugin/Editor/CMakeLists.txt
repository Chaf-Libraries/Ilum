function(add_widget name)
    set(TARGET_NAME Editor.${name})
    add_library(${TARGET_NAME} MODULE ${name}.cpp)
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Plugin/Editor")
    target_link_libraries(${TARGET_NAME} PRIVATE Core)
    target_link_libraries(${TARGET_NAME} PRIVATE Editor)
    target_link_libraries(${TARGET_NAME} PRIVATE Renderer)
    target_link_libraries(${TARGET_NAME} PRIVATE RHI)
    target_link_libraries(${TARGET_NAME} PRIVATE imgui)
    add_dependencies(Plugins ${TARGET_NAME})
    target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")
    copy_shared_lib(${TARGET_NAME} Editor)
endfunction()

add_widget(SceneView)
target_link_libraries(Editor.SceneView PRIVATE Resource Scene glfw)

add_widget(SceneHierarchy)
target_link_libraries(Editor.SceneHierarchy PRIVATE Scene)

add_widget(SceneInspector)
target_link_libraries(Editor.SceneInspector PRIVATE Scene)

add_widget(MainMenu)
target_link_libraries(Editor.MainMenu PRIVATE Scene nfd)

add_widget(MeshEditor)
target_link_libraries(Editor.MeshEditor PRIVATE Resource Geometry)

add_widget(AnimationEditor)
target_link_libraries(Editor.AnimationEditor PRIVATE nfd Resource Geometry)

add_widget(ResourceBrowser)
target_link_libraries(Editor.ResourceBrowser PRIVATE nfd Resource)

add_widget(RenderGraphEditor)
target_link_libraries(Editor.RenderGraphEditor PRIVATE RenderGraph nfd)

add_widget(MaterialGraphEditor)
target_link_libraries(Editor.MaterialGraphEditor PRIVATE Material nfd Resource Geometry RenderGraph)

add_widget(ShaderToy)
target_link_libraries(Editor.ShaderToy)
