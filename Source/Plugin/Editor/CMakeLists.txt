function(add_widget name)
    set(TARGET_NAME Editor.${name})
    add_library(${TARGET_NAME} MODULE ${name}.cpp)
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Plugin/Editor")
    target_link_libraries(${TARGET_NAME} PRIVATE Core)
    target_link_libraries(${TARGET_NAME} PRIVATE Editor)
    add_dependencies(Plugins ${TARGET_NAME})
    target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")
endfunction()

# add_widget(SceneView)
# target_link_libraries(Editor.SceneView PRIVATE Renderer glm imgui ImGuizmo Resource Scene RHI Geometry glfw)
# 
add_widget(SceneHierarchy)
target_link_libraries(Editor.SceneHierarchy PRIVATE SceneGraph imgui nativefiledialog Components Renderer RHI)

add_widget(SceneInspector)
target_link_libraries(Editor.SceneInspector PRIVATE SceneGraph imgui nativefiledialog Components Renderer RHI)