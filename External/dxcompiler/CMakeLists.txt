add_library(dxcompiler ${CMAKE_CURRENT_SOURCE_DIR}/lib/x64/dxcompiler.lib)
set_target_properties(dxcompiler PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(dxcompiler INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/inc)