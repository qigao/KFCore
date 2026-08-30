cmake_minimum_required(VERSION 3.20)

project(SiftInstalledConsumer LANGUAGES CXX)

set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

if(NOT DEFINED KFCORE_EXPECTED_KFCORE_DIR OR NOT DEFINED KFCORE_EXPECT_SIFT_POPSIFT)
  message(FATAL_ERROR "Expected KFCore package path and SIFT feature flags are required")
endif()

find_package(KFCore CONFIG REQUIRED
  PATHS "${KFCORE_EXPECTED_KFCORE_DIR}"
  NO_DEFAULT_PATH)

file(REAL_PATH "${KFCORE_EXPECTED_KFCORE_DIR}" _expected_kfcore_dir)
file(REAL_PATH "${KFCore_DIR}" _resolved_kfcore_dir)
if(NOT _resolved_kfcore_dir STREQUAL _expected_kfcore_dir)
  message(FATAL_ERROR "Consumer resolved KFCore outside its install prefix")
endif()
if(NOT TARGET KFCore::image_processor OR NOT TARGET KFCore::sift)
  message(FATAL_ERROR "Installed KFCore package does not export the SIFT dependency chain")
endif()
if(KFCORE_EXPECT_SIFT_POPSIFT)
  if(NOT TARGET KFCore::sift_popsift)
    message(FATAL_ERROR "Installed KFCore package does not export enabled PopSift support")
  endif()
  add_executable(sift_popsift_installed_consumer popsift_main.cpp)
  target_link_libraries(sift_popsift_installed_consumer PRIVATE KFCore::sift_popsift)
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/popsift-consumer-target-file-$<CONFIG>.txt"
    CONTENT "$<TARGET_FILE:sift_popsift_installed_consumer>")
elseif(TARGET KFCore::sift_popsift)
  message(FATAL_ERROR "Installed KFCore package unexpectedly exports PopSift support")
endif()

add_executable(sift_installed_consumer main.cpp)
target_link_libraries(sift_installed_consumer PRIVATE KFCore::sift)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer-target-file-$<CONFIG>.txt"
  CONTENT "$<TARGET_FILE:sift_installed_consumer>")
