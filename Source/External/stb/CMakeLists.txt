add_library(stb STATIC stb_image.h stb_image_write.h implementation.cpp)

target_include_directories(stb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})