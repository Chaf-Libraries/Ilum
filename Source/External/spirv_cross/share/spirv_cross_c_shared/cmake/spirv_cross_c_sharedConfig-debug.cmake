#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "spirv-cross-c-shared" for configuration "Debug"
set_property(TARGET spirv-cross-c-shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(spirv-cross-c-shared PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/spirv-cross-c-sharedd.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/spirv-cross-c-sharedd.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS spirv-cross-c-shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_spirv-cross-c-shared "${_IMPORT_PREFIX}/lib/spirv-cross-c-sharedd.lib" "${_IMPORT_PREFIX}/bin/spirv-cross-c-sharedd.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
