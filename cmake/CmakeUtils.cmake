# KFCore CMake utilities

include(GNUInstallDirs)

if(NOT DEFINED KFCORE_INSTALL_PLUGINDIR)
    set(KFCORE_INSTALL_PLUGINDIR "plugins" CACHE STRING
        "Install directory, relative to CMAKE_INSTALL_PREFIX, for KFCore execution plugins")
endif()

function(cmake_config_target target_name)
    set(options NO_INSTALL PLUGIN)
    set(oneValueArgs FOLDER VERSION SOVERSION EXPORT_NAME ALIAS EXPORT_SET)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_EXPORT_SET)
        set(ARG_EXPORT_SET KFCoreTargets)
    endif()

    if(ARG_ALIAS)
        add_library(${ARG_ALIAS} ALIAS ${target_name})
    endif()

    if(ARG_FOLDER)
        set_target_properties(${target_name} PROPERTIES FOLDER ${ARG_FOLDER})
    endif()

    if(ARG_EXPORT_NAME)
        set_target_properties(${target_name} PROPERTIES EXPORT_NAME ${ARG_EXPORT_NAME})
    endif()

    get_target_property(target_type ${target_name} TYPE)
    if(target_type STREQUAL "SHARED_LIBRARY")
        if(NOT ARG_VERSION AND PROJECT_VERSION)
            set(ARG_VERSION ${PROJECT_VERSION})
        endif()
        if(NOT ARG_SOVERSION AND PROJECT_VERSION_MAJOR)
            set(ARG_SOVERSION ${PROJECT_VERSION_MAJOR})
        endif()
        
        if(ARG_VERSION)
          set_target_properties(${target_name} PROPERTIES VERSION ${ARG_VERSION})
        endif()
        if(ARG_SOVERSION)
          set_target_properties(${target_name} PROPERTIES SOVERSION ${ARG_SOVERSION})
        endif()
    endif()

    if(NOT ARG_NO_INSTALL)
        if(target_type STREQUAL "INTERFACE_LIBRARY")
            install(TARGETS ${target_name}
                EXPORT ${ARG_EXPORT_SET})
        elseif(ARG_PLUGIN)
            if(UNIX AND NOT APPLE)
                target_link_options(${target_name} PRIVATE "LINKER:--no-as-needed")
            endif()
            install(TARGETS ${target_name}
                LIBRARY DESTINATION ${KFCORE_INSTALL_PLUGINDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${KFCORE_INSTALL_PLUGINDIR})
        else()
            install(TARGETS ${target_name}
                EXPORT ${ARG_EXPORT_SET}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
        endif()
    endif()
endfunction()

function(cmake_configure_target target_name)
    cmake_config_target(${target_name} ${ARGN})
endfunction()

function(cmake_install_headers)
    set(options)
    set(oneValueArgs DIRECTORY DESTINATION)
    set(multiValueArgs PATTERNS EXCLUDES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(ARG_DIRECTORY)
        set(match_args FILES_MATCHING PATTERN "*.h")
        foreach(p ${ARG_PATTERNS})
            list(APPEND match_args PATTERN "${p}")
        endforeach()
        foreach(e ${ARG_EXCLUDES})
            list(APPEND match_args PATTERN "${e}" EXCLUDE)
        endforeach()

        if(NOT ARG_DESTINATION)
            set(ARG_DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
        endif()

        install(DIRECTORY ${ARG_DIRECTORY}
            DESTINATION ${ARG_DESTINATION}
            ${match_args}
        )
    endif()
endfunction()

function(cmake_add_grammar TARGET_NAME)
  set(options)
  set(oneValueArgs LEXER_RE GRAMMAR_Y)
  set(multiValueArgs)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  string(TOLOWER "${TARGET_NAME}" target_name_lower)

  if(ARG_LEXER_RE)
    set(LEXER_GEN "${CMAKE_CURRENT_BINARY_DIR}/${target_name_lower}_lexer_gen.c")
    add_custom_command(
      OUTPUT ${LEXER_GEN}
      COMMAND ${RE2C_EXECUTABLE} -o ${LEXER_GEN} ${ARG_LEXER_RE}
      DEPENDS ${ARG_LEXER_RE}
      COMMENT "Generating ${TARGET_NAME} lexer with re2c"
      VERBATIM)
    set(LEXER_TARGET "${TARGET_NAME}_lexer_codegen")
    add_custom_target(${LEXER_TARGET} DEPENDS ${LEXER_GEN})
    set(${TARGET_NAME}_LEXER_GEN ${LEXER_GEN} PARENT_SCOPE)
    set(${TARGET_NAME}_LEXER_TARGET ${LEXER_TARGET} PARENT_SCOPE)
  endif()

  if(ARG_GRAMMAR_Y)
    set(GRAMMAR_GEN "${CMAKE_CURRENT_BINARY_DIR}/${target_name_lower}_grammar_gen.c")
    set(GRAMMAR_H "${CMAKE_CURRENT_BINARY_DIR}/${target_name_lower}_grammar_gen.h")
    set(GRAMMAR_Y_GEN "${CMAKE_CURRENT_BINARY_DIR}/${target_name_lower}_grammar_gen.y")
    add_custom_command(
      OUTPUT ${GRAMMAR_GEN} ${GRAMMAR_H}
      COMMAND ${CMAKE_COMMAND} -E copy ${ARG_GRAMMAR_Y} ${GRAMMAR_Y_GEN}
      COMMAND ${LEMON_EXECUTABLE} -T${LEMPAR} ${GRAMMAR_Y_GEN}
      DEPENDS ${ARG_GRAMMAR_Y} ${LEMON_DEPENDS}
      COMMENT "Generating ${TARGET_NAME} parser with lemon"
      VERBATIM)
    set(GRAMMAR_TARGET "${TARGET_NAME}_grammar_codegen")
    add_custom_target(${GRAMMAR_TARGET} DEPENDS ${GRAMMAR_GEN} ${GRAMMAR_H})
    set(${TARGET_NAME}_GRAMMAR_GEN ${GRAMMAR_GEN} PARENT_SCOPE)
    set(${TARGET_NAME}_GRAMMAR_H ${GRAMMAR_H} PARENT_SCOPE)
    set(${TARGET_NAME}_GRAMMAR_TARGET ${GRAMMAR_TARGET} PARENT_SCOPE)
  endif()
endfunction()

function(cmake_add_source VAR)
  set(options RECURSE)
  set(oneValueArgs)
  set(multiValueArgs DIRS EXCLUDES PATTERNS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(glob_mode GLOB)
  if(ARG_RECURSE)
    set(glob_mode GLOB_RECURSE)
  endif()

  if(NOT ARG_DIRS)
    set(ARG_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include")
  endif()

  if(NOT ARG_PATTERNS)
    set(ARG_PATTERNS "*.c" "*.cpp" "*.h" "*.hpp" "*.cc" "*.hh")
  endif()

  set(patterns)
  foreach(dir ${ARG_DIRS})
    foreach(pat ${ARG_PATTERNS})
      list(APPEND patterns "${dir}/${pat}")
    endforeach()
  endforeach()

  file(${glob_mode} collected ${patterns})

  if(ARG_EXCLUDES)
    list(REMOVE_ITEM collected ${ARG_EXCLUDES})
  endif()

  set(${VAR} ${collected} PARENT_SCOPE)
endfunction()

function(cmake_collect_files VAR)
  cmake_add_source(collected_files ${ARGN})
  set(${VAR} ${collected_files} PARENT_SCOPE)
endfunction()

function(cmake_add_test)
  set(options NO_TEST)
  set(oneValueArgs NAME TEST_NAME FOLDER)
  set(multiValueArgs
    SOURCES
    LIBS
    DEFS
    INCLUDES
    COMPILE_FEATURES
    TARGET_PROPERTIES
    TEST_COMMAND
    TEST_PROPERTIES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(ARG_NAME)
    if(NOT ARG_SOURCES)
      message(FATAL_ERROR "cmake_add_test(${ARG_NAME}) requires SOURCES")
    endif()
    if(TARGET ${ARG_NAME})
      message(FATAL_ERROR "cmake_add_test target already exists: ${ARG_NAME}")
    endif()

    add_executable(${ARG_NAME} ${ARG_SOURCES})
    if(ARG_LIBS)
      target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LIBS})
    endif()
    if(ARG_DEFS)
      target_compile_definitions(${ARG_NAME} PRIVATE ${ARG_DEFS})
    endif()
    if(ARG_INCLUDES)
      target_include_directories(${ARG_NAME} PRIVATE ${ARG_INCLUDES})
    endif()
    if(ARG_COMPILE_FEATURES)
      target_compile_features(${ARG_NAME} PRIVATE ${ARG_COMPILE_FEATURES})
    endif()
    if(ARG_TARGET_PROPERTIES)
      set_target_properties(${ARG_NAME} PROPERTIES ${ARG_TARGET_PROPERTIES})
    endif()
    cmake_config_target(${ARG_NAME} NO_INSTALL FOLDER "${ARG_FOLDER}")

    if(ARG_NO_TEST)
      return()
    endif()

    if(ARG_TEST_NAME)
      set(test_name ${ARG_TEST_NAME})
    else()
      set(test_name ${ARG_NAME})
    endif()
    if(ARG_TEST_COMMAND)
      add_test(NAME ${test_name} COMMAND ${ARG_TEST_COMMAND})
    else()
      add_test(NAME ${test_name} COMMAND ${ARG_NAME})
    endif()
    if(ARG_TEST_PROPERTIES)
      set_tests_properties(${test_name} PROPERTIES ${ARG_TEST_PROPERTIES})
    endif()
    return()
  endif()

  foreach(src ${ARG_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    if(NOT TARGET ${name})
      add_executable(${name} ${src})
      if(ARG_LIBS)
        target_link_libraries(${name} PRIVATE ${ARG_LIBS})
      endif()
      if(ARG_DEFS)
        target_compile_definitions(${name} PRIVATE ${ARG_DEFS})
      endif()
      if(ARG_INCLUDES)
        target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
      endif()
      if(ARG_COMPILE_FEATURES)
        target_compile_features(${name} PRIVATE ${ARG_COMPILE_FEATURES})
      endif()
      if(ARG_TARGET_PROPERTIES)
        set_target_properties(${name} PROPERTIES ${ARG_TARGET_PROPERTIES})
      endif()
      add_test(NAME ${name} COMMAND ${name})
      cmake_config_target(${name} NO_INSTALL FOLDER "${ARG_FOLDER}")
    endif()
  endforeach()
endfunction()

function(cmake_add_benchmark)
  set(options)
  set(oneValueArgs NAME FOLDER)
  set(multiValueArgs
    SOURCES
    LIBS
    DEFS
    INCLUDES
    COMPILE_FEATURES
    TARGET_PROPERTIES)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(ARG_NAME)
    if(NOT ARG_SOURCES)
      message(FATAL_ERROR "cmake_add_benchmark(${ARG_NAME}) requires SOURCES")
    endif()
    if(TARGET ${ARG_NAME})
      message(FATAL_ERROR "cmake_add_benchmark target already exists: ${ARG_NAME}")
    endif()

    add_executable(${ARG_NAME} ${ARG_SOURCES})
    if(ARG_LIBS)
      target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LIBS})
    endif()
    if(ARG_DEFS)
      target_compile_definitions(${ARG_NAME} PRIVATE ${ARG_DEFS})
    endif()
    if(ARG_INCLUDES)
      target_include_directories(${ARG_NAME} PRIVATE ${ARG_INCLUDES})
    endif()
    if(ARG_COMPILE_FEATURES)
      target_compile_features(${ARG_NAME} PRIVATE ${ARG_COMPILE_FEATURES})
    endif()
    if(ARG_TARGET_PROPERTIES)
      set_target_properties(${ARG_NAME} PROPERTIES ${ARG_TARGET_PROPERTIES})
    endif()
    cmake_config_target(${ARG_NAME} NO_INSTALL FOLDER "${ARG_FOLDER}")
    return()
  endif()

  foreach(src ${ARG_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    if(NOT TARGET ${name})
      add_executable(${name} ${src})
      if(ARG_LIBS)
        target_link_libraries(${name} PRIVATE ${ARG_LIBS})
      endif()
      if(ARG_DEFS)
        target_compile_definitions(${name} PRIVATE ${ARG_DEFS})
      endif()
      if(ARG_INCLUDES)
        target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
      endif()
      if(ARG_COMPILE_FEATURES)
        target_compile_features(${name} PRIVATE ${ARG_COMPILE_FEATURES})
      endif()
      if(ARG_TARGET_PROPERTIES)
        set_target_properties(${name} PROPERTIES ${ARG_TARGET_PROPERTIES})
      endif()
      cmake_config_target(${name} NO_INSTALL FOLDER "${ARG_FOLDER}")
    endif()
  endforeach()
endfunction()
