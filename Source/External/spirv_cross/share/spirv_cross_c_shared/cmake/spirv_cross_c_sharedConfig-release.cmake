#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "spirv-cross-c-shared" for configuration "Release"
set_property(TARGET spirv-cross-c-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(spirv-cross-c-shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/spirv-cross-c-shared.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/spirv-cross-c-shared.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS spirv-cross-c-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_spirv-cross-c-shared "${_IMPORT_PREFIX}/lib/spirv-cross-c-shared.lib" "${_IMPORT_PREFIX}/bin/spirv-cross-c-shared.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
