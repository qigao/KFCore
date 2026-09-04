add_library(kfcore_sift SHARED src/sift_extractor.cpp)
target_include_directories(kfcore_sift PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_sift PUBLIC cxx_std_17)
target_link_libraries(kfcore_sift PUBLIC KFCore::image_processor_core)
set_target_properties(kfcore_sift PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "sift"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_sift
  ALIAS KFCore::sift
  EXPORT_NAME sift
  EXPORT_SET KFCoreTargets
  FOLDER "sift")

if(KFCORE_ENABLE_CUDA)
  add_subdirectory(vendor/popsift)

  add_library(kfcore_sift_popsift SHARED src/popsift_extractor.cpp)
  target_include_directories(kfcore_sift_popsift PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_compile_features(kfcore_sift_popsift PUBLIC cxx_std_17)
  target_link_libraries(kfcore_sift_popsift
    PUBLIC KFCore::sift
    PRIVATE KFCore::image_processor_cuda kfcore_popsift_internal CUDA::cudart)
  set_target_properties(kfcore_sift_popsift PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "sift"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
  cmake_config_target(kfcore_sift_popsift
    ALIAS KFCore::sift_popsift
    EXPORT_NAME sift_popsift
    EXPORT_SET KFCoreTargets
    FOLDER "sift")
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/popsift/COPYING.md"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/KFCore/popsift")
  install(SCRIPT "${KFCORE_POPSIFT_REVISION_VERIFY_SCRIPT}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vendor/popsift/generated/UPSTREAM.md"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/KFCore/popsift")
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
