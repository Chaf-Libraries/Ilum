message(STATUS "include Build.cmake successfully")



# add subdirectories(relative)
function(AddSubDirRec path)
    file(GLOB_RECURSE subpath LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/${path}/*)
    set(dirs "")
    list(APPEND subpath ${CMAKE_CURRENT_SOURCE_DIR}/${path})
    foreach(item ${subpath})
        if(IS_DIRECTORY ${item} AND EXISTS "${item}/CMakeLists.txt")
            list(APPEND dirs ${item})
        endif()
    endforeach()
    foreach(dir ${dirs})
        add_subdirectory(${dir})
    endforeach()
endfunction()

# get target name
function(GetTargetName TargetName _Path)
    file(RELATIVE_PATH TargetRelPath "${PROJECT_SOURCE_DIR}" "${_Path}")
    STRING(REGEX REPLACE ".*/(.*)" "\\1" TargetRelPath ${TargetRelPath} )
    string(REPLACE "/" "_" tmp "${PROJECT_NAME}_${TargetRelPath}")
    set(${TargetName} ${tmp} PARENT_SCOPE)
endfunction()

# expand source files
function(ExpandSrc res src)
    set(tmp_res "")
    foreach(item ${${src}})
        if(IS_DIRECTORY ${item})
            file(GLOB_RECURSE itemSrc
                # cmake
                ${item}/*.cmake

                # msvc
                ${item}/*.natvis

                # header
                ${item}/*.h
                ${item}/*.hpp
                ${item}/*.hxx
                ${item}/*.inl
            
                # source
                ${item}/*.c
                ${item}/*.cpp
                ${item}/*.cxx
                ${item}/*.cc

                # opengl shader
                ${item}/*.glsl
                ${item}/*.vert
                ${item}/*.frag
                ${item}/*.tesc
                ${item}/*.tese
                ${item}/*.geom
                ${item}/*.comp

                # ...
            )
            list(APPEND tmp_res ${itemSrc})
        else()
            if(NOT IS_ABSOLUTE "${item}")
                get_filename_component(item "${item}" ABSOLUTE)
            endif()
            list(APPEND tmp_res ${item})
        endif()
    endforeach()
    set(${res} ${tmp_res} PARENT_SCOPE)  
endfunction()

# set target
function(SetTarget)
    set(arglist "")

    # public
    list(APPEND arglist SOURCE_PUBLIC INC LIB DEFINE C_OPTION L_OPTION)
    # private
    list(APPEND arglist SOURCE INC_PRIVATE LIB_PRIVATE DEFINE_PRIVATE C_OPTION_PRIVATE L_OPTION_PRIVATE)
    # interface
    list(APPEND arglist SOURCE_INTERFACE INC_INTERFACE LIB_INTERFACE DEFINE_INTERFACE C_OPTION_INTERFACE L_OPTION_INTERFACE)

    cmake_parse_arguments("ARG" "TEST;NO_GROUP" "MODE;FEATURE;TARGET_NAME" "${arglist}" ${ARGN})

    # parameters
    # TEST
    # NO_GROUP: default no group
    # MDOE: EXE/STATIC/SHARED/INTERFACE
    # FEATURE: PUBLIC/INTERFACE/PRIVATE(default)/NONE
    # TARGET_NAME: (default)/reset
    # [arglist]: public, interface, private
    # SOURCE: dir(recursive), files, auto add current dir   |   target_sources
    # INC: dir                                              |   target_include_directories
    # LIB: <lib-target>, *.lib                              |   target_link_libraries
    # DEFINE: #define...                                    |   target_compile_definitions
    # C_OPTION: compile options                             |   target_compile_options
    # L_OPTION: link options                                |   target_link_options                            

    # default
    if("${ARG_FEATURE}" STREQUAL "")
        set(ARG_FEATURE "PRIVATE")
    endif()

    # source
    if("${ARG_FEATURE}" STREQUAL "PUBLIC")
        list(APPEND ARG_SOURCE_PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    elseif("${ARG_FEATURE}" STREQUAL "PRIVATE")
        list(APPEND ARG_SOURCE_PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    elseif("${ARG_FEATURE}" STREQUAL "INTERFACE")
        list(APPEND ARG_SOURCE_INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    else()
        message(FATAL_ERROR "unsupported feature ${ARG_FEATURE}!")
    endif()

    # interface = public + private
    if("${ARG_MODE}" STREQUAL "INTERFACE")
        list(APPEND ARG_SOURCE_INTERFACE ${ARG_SOURCE_PUBLIC} ${ARG_SOURCE})
        list(APPEND ARG_INC_INTERFACE ${ARG_INC} ${ARG_INC_PRIVATE})
        list(APPEND ARG_LIB_INTERFACE ${ARG_LIB} ${ARG_LIB_PRIVATE})
        list(APPEND ARG_DEFINE_INTERFACE ${ARG_DEFINE} ${ARG_DEFINE_PRIVATE})
        list(APPEND ARG_C_OPTION_INTERFACE ${ARG_C_OPTION} ${ARG_C_OPTION_PRIVATE})
        list(APPEND ARG_L_OPTION_INTERFACE ${ARG_L_OPTION} ${ARG_L_OPTION_PRIVATE})

        set(ARG_SOURCE_PUBLIC "")
        set(ARG_SOURCE "")
        set(ARG_INC "")
        set(ARG_INC_PRIVATE "")
        set(ARG_LIB "")
        set(ARG_LIB_PRIVATE "")
        set(ARG_DEFINE "")
        set(ARG_DEFINE_PRIVATE "")
        set(ARG_C_OPTION "")
        set(ARG_C_OPTION_PRIVATE "")
        set(ARG_L_OPTION "")
        set(ARG_L_OPTION_PRIVATE "")

        set(ARG_FEATURE "INTERFACE")
    endif()

    ExpandSrc(src_public ARG_SOURCE_PUBLIC)
    ExpandSrc(src_private ARG_SOURCE_PRIVATE)
    ExpandSrc(src_interface ARG_SOURCE_INTERFACE)

    # test
    if(ARG_TEST AND NOT "${Build_Test_${PROJECT_NAME}}")
        return()
    endif()

    # group
    if(NOT NO_GROUP)
        set(sources ${src_public} ${src_private} ${src_interface})
        foreach(src ${sources})
            get_filename_component(dir ${src} DIRECTORY)
            string(FIND ${dir} ${CMAKE_CURRENT_SOURCE_DIR} idx)
            if(NOT idx EQUAL -1)
                set(base_dir "${CMAKE_CURRENT_SOURCE_DIR}/..")
            else()
                set(base_dir ${PROJECT_SOURCE_DIR})
            endif()
            file(RELATIVE_PATH rdir ${base_dir} ${dir})
            if(MSVC)
                string(REPLACE "/" "\\" rdir ${rdir})
            endif()
            source_group(${rdir} FILES ${src})
        endforeach()
    endif()

    # target folder
    file(RELATIVE_PATH targerRelPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
    
    set(targetFolder "${PROJECT_NAME}/${targerRelPath}")
    # target name
    GetTargetName(targetName ${CMAKE_CURRENT_SOURCE_DIR})
    if(NOT "${ARG_TARGET_NAME}" STREQUAL "")
        set(targetName ${ARG_TARGET_NAME})
    endif()

    # print
    message(STATUS "")
    message(STATUS "---Setting Target---")
    message(STATUS " - name: ${targetName}")
    message(STATUS " - folder: ${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS " - mode: ${ARG_MODE}")

    PrintList(
        TITLE " - sources private: "
        PREFIX "    "
        STR "${src_private}"
    )
    PrintList(
        TITLE " - sources public: "
        PREFIX "    "
        STR "${src_public}"
    )
    PrintList(
        TITLE " - sources interface: "
        PREFIX "    "
        STR "${src_interface}"
    )
    PrintList(
        TITLE " - define public: "
        PREFIX "    "
        STR "${ARG_DEFINE}"
    )
    PrintList(
        TITLE " - define private: "
        PREFIX "    "
        STR "${ARG_DEFINE_PRIVATE}"
    )
    PrintList(
        TITLE " - define interface: "
        PREFIX "    "
        STR "${ARG_DEFINE_INTERFACE}"
    )
    PrintList(
        TITLE " - lib public: "
        PREFIX "    "
        STR "${ARG_LIB}"
    )
    PrintList(
        TITLE " - lib private: "
        PREFIX "    "
        STR "${ARG_LIB_PRIVATE}"
    )
    PrintList(
        TITLE " - lib interface: "
        PREFIX "    "
        STR "${ARG_LIB_INTERFACE}"
    )
    PrintList(
        TITLE " - inc public: "
        PREFIX "    "
        STR "${ARG_INC}"
    )
    PrintList(
        TITLE " - inc private: "
        PREFIX "    "
        STR "${ARG_INC_PRIVATE}"
    )
    PrintList(
        TITLE " - inc interface: "
        PREFIX "    "
        STR "${ARG_INC_INTERFACE}"
    )
    PrintList(
        TITLE " - compile option public: "
        PREFIX "    "
        STR "${ARG_C_OPTION}"
    )
    PrintList(
        TITLE " - compile option private: "
        PREFIX "    "
        STR "${ARG_C_OPTION_PRIVATE}"
    )
    PrintList(
        TITLE " - compile option interface: "
        PREFIX "    "
        STR "${ARG_C_OPTION_INTERFACE}"
    )
    PrintList(
        TITLE " - link option public: "
        PREFIX "    "
        STR "${ARG_L_OPTION}"
    )
    PrintList(
        TITLE " - link option private: "
        PREFIX "    "
        STR "${ARG_L_OPTION_PRIVATE}"
    )
    PrintList(
        TITLE " - link option interface: "
        PREFIX "    "
        STR "${ARG_L_OPTION_INTERFACE}"
    )

    PackageName(package_name)

    # add target
    if("${ARG_MODE}" STREQUAL "EXE")
        add_executable(${targetName})
        add_executable("Chaf::${targetName}" ALIAS ${targetName})
        if(MSVC)
            set_target_properties(${targetName} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
        endif()
        set_target_properties(${targetName} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    elseif("${ARG_MODE}" STREQUAL "STATIC")
        add_library(${targetName} STATIC)
        add_library("Chaf::${targetName}" ALIAS ${targetName})
    elseif("${ARG_MODE}" STREQUAL "SHARED")
        add_library(${targetName} SHARED)
        add_library("Chaf::${targetName}" ALIAS ${targetName})
    elseif("${ARG_MODE}" STREQUAL "INTERFACE")
        add_library(${targetName} INTERFACE)
        add_library("Chaf::${targetName}" ALIAS ${targetName})
    else()
        message(FATAL_ERROR "Unsupported mode: ${ATG_MODE}")
        return()
    endif()

    # folder
    set_target_properties(${targetName} PROPERTIES FOLDER ${targetFolder})

    # target source
    foreach(src ${src_public})
        get_filename_component(abs_src ${src} ABSOLUTE)
        file(RELATIVE_PATH rel_src ${PROJECT_SOURCE_DIR} ${abs_src})
        target_sources(
            ${targetName} PUBLIC
            $<BUILD_INTERFACE:${abs_src}>
            $<INSTALL_INTERFACE:${package_name}/${rel_src}>
        )
    endforeach()
    foreach(src ${src_private})
        get_filename_component(abs_src ${src} ABSOLUTE)
        file(RELATIVE_PATH rel_src ${PROJECT_SOURCE_DIR} ${abs_src})
        target_sources(
            ${targetName} PRIVATE
            $<BUILD_INTERFACE:${abs_src}>
            $<INSTALL_INTERFACE:${package_name}/${rel_src}>
        )
    endforeach()
    foreach(src ${src_interface})
        get_filename_component(abs_src ${src} ABSOLUTE)
        file(RELATIVE_PATH rel_src ${PROJECT_SOURCE_DIR} ${abs_src})
        target_sources(
            ${targetName} INTERFACE
            $<BUILD_INTERFACE:${abs_src}>
            $<INSTALL_INTERFACE:${package_name}/${rel_src}>
        )
    endforeach()

    # target define
    target_compile_definitions(
        ${targetName}
        PUBLIC ${ARG_DEFINE}
        INTERFACE ${ARG_DEFINE_INTERFACE}
        PRIVATE ${ARG_DEFINE_PRIVATE}
    )

    # target lib
    target_link_libraries(
        ${targetName}
        PUBLIC ${ARG_LIB}
        INTERFACE ${ARG_LIB_INTERFACE}
        PRIVATE ${ARG_LIB_PRIVATE}
    )

    # target inc
    foreach(inc ${ARG_INC})
        get_filename_component(abs_inc ${inc} ABSOLUTE)
        file(RELATIVE_PATH rel_inc ${PROJECT_SOURCE_DIR} ${abs_inc})
        target_include_directories(
            ${targetName} PUBLIC
            $<BUILD_INTERFACE:${abs_inc}>
            $<INSTALL_INTERFACE:${package_name}/${rel_inc}>
        )
    endforeach()
    foreach(inc ${ARG_INC_PRIVATE})
        get_filename_component(abs_inc ${inc} ABSOLUTE)
        file(RELATIVE_PATH rel_inc ${PROJECT_SOURCE_DIR} ${abs_inc})
        target_include_directories(
            ${targetName} PRIVATE
            $<BUILD_INTERFACE:${abs_inc}>
            $<INSTALL_INTERFACE:${package_name}/${rel_inc}>
        )
    endforeach()
    foreach(inc ${ARG_INC_INTERFACE})
        get_filename_component(abs_inc ${inc} ABSOLUTE)
        file(RELATIVE_PATH rel_inc ${PROJECT_SOURCE_DIR} ${abs_inc})
        target_include_directories(
            ${targetName} INTERFACE
            $<BUILD_INTERFACE:${abs_inc}>
            $<INSTALL_INTERFACE:${package_name}/${rel_inc}>
        )
    endforeach()

    # target compile option
    target_compile_options(
        ${targetName}
        PUBLIC ${ARG_C_OPTION}
        INTERFACE ${ARG_C_OPTION_INTERFACE}
        PRIVATE ${ARG_C_OPTION_PRIVATE}
    )

    # terget link option
    target_link_options(
        ${targetName}
        PUBLIC ${ARG_L_OPTION}
        INTERFACE ${ARG_L_OPTION_INTERFACE}
        PRIVATE ${ARG_L_OPTION_PRIVATE}
    )

    if(NOT ARG_TEST)
        install(
            TARGETS ${targetName}
            EXPORT "${PROJECT_NAME}Targets"
            RUNTIME DESTINATION "bin"
            ARCHIVE DESTINATION "${package_name}/lib"
            LIBRARY DESTINATION "${package_name}/lib"
        )
    endif()

    message(STATUS "--------------------")
endfunction()