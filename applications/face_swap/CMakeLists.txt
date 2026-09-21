add_library(kfcore_face_swap_application STATIC
  src/application.cpp
  src/composer.cpp
  src/geometry.cpp
  src/preprocess.cpp)
target_include_directories(kfcore_face_swap_application
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_face_swap_application PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_swap_application
  PUBLIC
    KFCore::face_model_runtime
    KFCore::runtime_core
  PRIVATE
    KFCore::image_processor_cpu)
set_target_properties(kfcore_face_swap_application PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "applications/face_swap")
cmake_config_target(kfcore_face_swap_application
  ALIAS KFCore::face_swap_application
  EXPORT_NAME face_swap_application
  EXPORT_SET KFCoreTargets
  FOLDER "applications/face_swap")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
