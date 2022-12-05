function(add_importer name)
    set(TARGET_NAME Importer.${name})
    add_library(${TARGET_NAME} MODULE ${name}.cpp)
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Plugin/Importer")
    target_link_libraries(${TARGET_NAME} PRIVATE Core)
    target_link_libraries(${TARGET_NAME} PRIVATE RHI)
    target_link_libraries(${TARGET_NAME} PRIVATE Resource)
    add_dependencies(Plugins ${TARGET_NAME})
    target_compile_definitions(${TARGET_NAME} PRIVATE "-DNOMINMAX")
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:MSVCRT")
endfunction()

add_importer(Assimp.Model)
target_link_libraries(Importer.Assimp.Model PRIVATE assimp::assimp Geometry meshoptimizer::meshoptimizer Animation)

# add_importer(Assimp.Animation)
# target_link_libraries(Importer.Assimp.Animation PRIVATE assimp::assimp Animation)

add_importer(DDS)

add_importer(STB)
target_link_libraries(Importer.STB PRIVATE stb)

# add_importer(CGLTF)
# target_include_directories(Importer.CGLTF PRIVATE ${PROJECT_SOURCE_DIR}/Source/External/cgltf)