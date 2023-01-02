

function(add_material_node category name)
    set(TARGET_NAME Material.${category}.${name})
    add_library(${TARGET_NAME} MODULE ${name}.cpp)
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Plugin/Material/${category}")
    target_link_libraries(${TARGET_NAME} PRIVATE Core)
    target_link_libraries(${TARGET_NAME} PRIVATE RHI)
    target_link_libraries(${TARGET_NAME} PRIVATE Material)
    target_link_libraries(${TARGET_NAME} PRIVATE Resource)
    target_link_libraries(${TARGET_NAME} PRIVATE imgui)
    target_link_libraries(${TARGET_NAME} PRIVATE Editor)
    target_link_libraries(${TARGET_NAME} PRIVATE Renderer)
    add_dependencies(Plugins ${TARGET_NAME})
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/Source/Plugin/MaterialNode)
    target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")
    copy_shared_lib(${TARGET_NAME} Material ${TARGET_NAME}.dll)
endfunction()



add_subdirectory(Input)
add_subdirectory(Output)
add_subdirectory(Converter)
add_subdirectory(Texture)
add_subdirectory(BSDF)