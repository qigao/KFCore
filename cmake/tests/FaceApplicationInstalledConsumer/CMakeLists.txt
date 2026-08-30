cmake_minimum_required(VERSION 3.20)

project(FaceApplicationInstalledConsumer LANGUAGES CXX)

set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

if(NOT DEFINED KFCORE_EXPECTED_KFCORE_DIR)
  message(FATAL_ERROR "Expected KFCore directory is required")
endif()

find_package(KFCore CONFIG REQUIRED
  PATHS "${KFCORE_EXPECTED_KFCORE_DIR}"
  NO_DEFAULT_PATH)

file(REAL_PATH "${KFCORE_EXPECTED_KFCORE_DIR}" _expected_kfcore_dir)
file(REAL_PATH "${KFCore_DIR}" _resolved_kfcore_dir)
if(NOT _resolved_kfcore_dir STREQUAL _expected_kfcore_dir)
  message(FATAL_ERROR "Consumer resolved KFCore outside its install prefix")
endif()
if(NOT TARGET KFCore::face_applications)
  message(FATAL_ERROR "Installed KFCore package does not export KFCore::face_applications")
endif()

add_executable(face_application_installed_consumer main.cpp)
target_link_libraries(face_application_installed_consumer PRIVATE
  KFCore::face_applications)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-target-file-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:face_application_installed_consumer>")
