cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

project(
  KFCore
  VERSION 0.1.0
  DESCRIPTION "Kalman filter implementations in C"
  HOMEPAGE_URL "https://github.com/qigao/KFCore"
  LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")
include(CMakeOptions.cmake)
include(GNUInstallDirs)
include(Sanitizers)
include(CmakeUtils)

find_package(Threads REQUIRED)
find_package(TurboNet CONFIG REQUIRED PATHS "${TURBONET_ROOT}")

include(CTest)

add_subdirectory(vendor)
add_subdirectory(kalman)
add_subdirectory(turbo_serial)

if(KFCORE_INSTALL)
  include(CMakePackageConfigHelpers)

  install(
    TARGETS kfcore  
    EXPORT KFCoreTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

  install(FILES kalman/kalman_takasu.h kalman/kalman_udu.h kalman/kalman_ekf.h
                kalman/kalman_ukf.h
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore")

  if(KFCORE_BUILD_NAVIGATION_TOOLS)
    install(FILES kalman/navigation_tools/navtoolbox.h
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
