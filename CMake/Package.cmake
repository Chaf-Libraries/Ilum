set(_${PROJECT_NAME}_have_dependencies 0)

include(FetchContent)

function(DecodeVersion major minor patch version)
  string(REGEX REPLACE "[^0-9 .]" "" version ${version})
  if("${version}" MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(${major} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    set(${minor} "${CMAKE_MATCH_2}" PARENT_SCOPE)
    set(${patch} "${CMAKE_MATCH_3}" PARENT_SCOPE)
  elseif("${version}" MATCHES "^([0-9]+)\\.([0-9]+)")
    set(${major} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    set(${minor} "${CMAKE_MATCH_2}" PARENT_SCOPE)
    set(${patch} "" PARENT_SCOPE)
  else()
    set(${major} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    set(${minor} "" PARENT_SCOPE)
    set(${patch} "" PARENT_SCOPE)
  endif()
endfunction()

function(ToPackageName rst name version)
  set(tmp "${name}_${version}")
  string(REPLACE "." "_" tmp ${tmp})
  set(${rst} "${tmp}" PARENT_SCOPE)
endfunction()

function(PackageName rst)
  ToPackageName(tmp ${PROJECT_NAME} ${PROJECT_VERSION})
  set(${rst} ${tmp} PARENT_SCOPE)
endfunction()

macro(AddDepPro projectName name version author)
  set(_${projectName}_have_dependencies 1)
  list(FIND _${projectName}_dep_name_list "${name}" _idx)
  if("${_idx}" STREQUAL "-1")
      message(STATUS "start add dependence ${name} ${version}")
    set(_need_find TRUE)
  else()
    set(_A_version "${${name}_VERSION}")
    set(_B_version "${version}")
    DecodeVersion(_A_major _A_minor _A_patch "${_A_version}")
    DecodeVersion(_B_major _B_minor _B_patch "${_B_version}")
    if("${_A_version}" STREQUAL "${_B_version}")
      message(STATUS "Diamond dependence of ${name} with same version: ${_A_version} and ${_B_version}")
      set(_need_find FALSE)
    else()
      message(FATAL_ERROR "Diamond dependence of ${name} with incompatible version: ${_A_version} and ${_B_version}")
    endif()
  endif()
  if("${_need_find}" STREQUAL TRUE)
    list(APPEND _${projectName}_dep_name_list ${name})
    list(APPEND _${projectName}_dep_version_list ${version})
    message(STATUS "find package: ${name} ${version}")
    string(REGEX REPLACE "[^0-9 .]" "" _version ${version})
    find_package(${name} ${_version} QUIET)
    if(${${name}_FOUND})
      message(STATUS "${name} ${${name}_VERSION} found")
    else()
      if("${author}" STREQUAL "")
        set(_address "https://github.com/Chaphlagical/${name}")
      else()
        set(_address "https://github.com/${author}/${name}")
      endif()
      
      message(STATUS "${name} ${version} not found")
      message(STATUS "fetch: ${_address} with tag ${version}")
      FetchContent_Declare(
        ${name}
        GIT_REPOSITORY ${_address}
        GIT_TAG "${version}"
      )
      FetchContent_MakeAvailable(${name})
      message(STATUS "${name} ${version} build done")
    endif()
  endif()
endmacro()

macro(AddDep name version author)
  AddDepPro(${PROJECT_NAME} ${name} ${version} ${author})
endmacro()

macro(ExportProject)
  cmake_parse_arguments("ARG" "" "TARGET" "DIRECTORIES" ${ARGN})
  
  PackageName(package_name)
  message(STATUS "export ${package_name}")

  set(PACKAGE_INIT "
get_filename_component(include_dir \"${CMAKE_CURRENT_LIST_DIR}/../include\" ABSOLUTE)
include_directories(\"${include_dir}\")\n")
  
  if(${_${PROJECT_NAME}_have_dependencies})
    set(PACKAGE_INIT "${PACKAGE_INIT}
if(NOT FetchContent_FOUND)
  include(FetchContent)
endif()
if(NOT cpkg_FOUND)
  message(STATUS \"find package: cpkg ${cpkg_VERSION}\")
  find_package(cpkg ${cpkg_VERSION} QUIET)
  if(NOT cpkg_FOUND)
    set(_${PROJECT_NAME}_address \"https://github.com/Chaphlagical/cpkg\")
    message(STATUS \"cpkg ${cpkg_VERSION} not found\")
    message(STATUS \"fetch: ${_${PROJECT_NAME}_address} with tag ${cpkg_VERSION}\")
    FetchContent_Declare(
      cpkg
      GIT_REPOSITORY ${_${PROJECT_NAME}_address}
      GIT_TAG ${cpkg_VERSION}
    )
    FetchContent_MakeAvailable(cpkg)
    message(STATUS \"cpkg ${cpkg_VERSION} build done\")
  endif()
endif()
"
        )

    message(STATUS "[Dependencies]")
    list(LENGTH _${PROJECT_NAME}_dep_name_list _${PROJECT_NAME}_dep_num)
    math(EXPR _${PROJECT_NAME}_stop "${_${PROJECT_NAME}_dep_num}-1")
    foreach(index RANGE ${_${PROJECT_NAME}_stop})
      list(GET _${PROJECT_NAME}_dep_name_list ${index} dep_name)
      list(GET _${PROJECT_NAME}_dep_version_list ${index} dep_version)
      message(STATUS "- ${dep_name} ${dep_version}")
      string(APPEND PACKAGE_INIT "AddDepPro(${PROJECT_NAME} ${dep_name} ${dep_version})\n")
    endforeach()
  endif()
  
  if(NOT "${ARG_TARGET}" STREQUAL "OFF")
    # generate the export targets for the build tree
    # needs to be after the install(TARGETS ) command
    export(EXPORT "${PROJECT_NAME}Targets"
      NAMESPACE "Chaf::"
      #FILE "CMAKECURRENTBINARYDIR/{PROJECT_NAME}Targets.cmake"
    )
    
    # install the configuration targets
    install(EXPORT "${PROJECT_NAME}Targets"
      FILE "${PROJECT_NAME}Targets.cmake"
      NAMESPACE "Chaf::"
      DESTINATION "${package_name}/cmake"
    )
  endif()
  
  include(CMakePackageConfigHelpers)
  # generate the config file that is includes the exports
  configure_package_config_file(${PROJECT_SOURCE_DIR}/config/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${package_name}/cmake"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )
  
  # generate the version file for the config file
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
  )

  # install the configuration file
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${package_name}/cmake"
  )
  
  foreach(dir ${ARG_DIRECTORIES})
    string(REGEX MATCH "(.*)/" prefix ${dir})
    if("${CMAKE_MATCH_1}" STREQUAL "")
      set(_destination "${package_name}")
    else()
      set(_destination "${package_name}/${CMAKE_MATCH_1}")
    endif()
    install(DIRECTORY ${dir} DESTINATION "${_destination}")
  endforeach()
endmacro()