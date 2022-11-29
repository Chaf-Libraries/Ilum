add_library(vulkan STATIC volk.h volk.c implementation.cpp)

target_include_directories(vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(vulkan PUBLIC VK_NO_PROTOTYPES)

if(ANDROID)
    target_compile_definitions(vulkan PUBLIC VK_USE_PLATFORM_ANDROID_KHR)
elseif(WIN32)
    target_compile_definitions(vulkan PUBLIC VK_USE_PLATFORM_WIN32_KHR)
elseif(APPLE)
    target_compile_definitions(vulkan PUBLIC VK_USE_PLATFORM_MACOS_MVK)
elseif(UNIX)
    # See whether X11 is available. If not, fall back to direct-to-display mode.
    find_package(X11 QUIET)
    if (X11_FOUND)
        target_compile_definitions(vulkan PUBLIC VK_USE_PLATFORM_XCB_KHR)
    else()
        set(DIRECT_TO_DISPLAY TRUE)
        set(DIRECT_TO_DISPLAY TRUE PARENT_SCOPE)
        target_compile_definitions(vulkan PUBLIC VK_USE_PLATFORM_DISPLAY_KHR)
    endif()
endif() 

target_compile_definitions(vulkan PUBLIC VMA_STATIC_VULKAN_FUNCTIONS)
