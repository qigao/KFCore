cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(
  KFCore
  VERSION 0.1.0
  DESCRIPTION "Kalman filter implementations in C"
  HOMEPAGE_URL "https://github.com/jnz/KFCore"
  LANGUAGES C)

include(CMakeOptions.cmake)
include(GNUInstallDirs)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")

set(KFCORE_PUBLIC_HEADERS c/kalman_takasu.h c/kalman_udu.h c/kalman_ekf.h
                          c/kalman_ukf.h c/linalg.h c/miniblas.h)

add_library(kfcore)
add_library(KFCore::kfcore ALIAS kfcore)

target_sources(
  kfcore
  PRIVATE c/kalman_takasu.c
          c/kalman_udu.c
          c/kalman_ekf.c
          c/kalman_ukf.c
          c/linalg.c
          c/miniblas.c
          ${KFCORE_PUBLIC_HEADERS})

target_include_directories(
  kfcore PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/c>"
                "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kfcore>")

target_compile_features(kfcore PUBLIC c_std_11)

if(KFCORE_USE_SIMDE)
  find_path(SIMDE_INCLUDE_DIR NAMES simde/x86/sse.h REQUIRED)
  target_include_directories(kfcore PRIVATE "${SIMDE_INCLUDE_DIR}")
  target_compile_definitions(kfcore PRIVATE KFCORE_USE_SIMDE=1)
endif()

if(KFCORE_BUILD_NAVIGATION_TOOLS)
  list(APPEND KFCORE_PUBLIC_HEADERS c/navigation_tools/navtoolbox.h)
  target_sources(kfcore PRIVATE c/navigation_tools/navtoolbox.c
                                c/navigation_tools/navtoolbox.h)
  target_include_directories(
    kfcore
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/c/navigation_tools>")
endif()

if(KFCORE_ENABLE_WARNINGS)
  target_compile_options(
    kfcore
    PRIVATE $<$<C_COMPILER_ID:MSVC>:/W4> $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall
            -Wextra -Wpedantic -Wdouble-promotion>)
endif()

if(NOT WIN32)
  target_link_libraries(kfcore PUBLIC m)
endif()

include(CTest)
if(ENABLE_TESTS)
  function(kfcore_add_test test_target)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS" ${ARGN})
    add_executable(${test_target} ${ARG_SOURCES})
    target_link_libraries(${test_target} PRIVATE KFCore::kfcore ${ARG_LIBS})
    add_test(NAME ${test_target} COMMAND ${test_target})
  endfunction()

  kfcore_add_test(kfcore_tests SOURCES tests/test.c)

endif()

if(KFCORE_INSTALL)
  include(CMakePackageConfigHelpers)

  install(
    TARGETS kfcore
    EXPORT KFCoreTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

  install(FILES c/kalman_takasu.h c/kalman_udu.h c/kalman_ekf.h c/kalman_ukf.h
                c/linalg.h c/miniblas.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore")

  if(KFCORE_BUILD_NAVIGATION_TOOLS)
    install(FILES c/navigation_tools/navtoolbox.h
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore/navigation_tools")
  endif()

  install(
    EXPORT KFCoreTargets
    NAMESPACE KFCore::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")

  configure_package_config_file(
    cmake/KFCoreConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/KFCoreConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/KFCoreConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/KFCoreConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/KFCoreConfigVersion.cmake"
          DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/KFCore")
endif()
