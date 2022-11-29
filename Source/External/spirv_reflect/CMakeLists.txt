add_library(spirv_reflect STATIC spirv_reflect.h spirv_reflect.c)

target_include_directories(spirv_reflect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})