add_library(kfcore_thig STATIC
  src/temporal_graph.cpp)
target_include_directories(kfcore_thig
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_thig PUBLIC cxx_std_17)
set_target_properties(kfcore_thig PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "thig")
cmake_config_target(kfcore_thig
  ALIAS KFCore::thig
  EXPORT_NAME thig
  EXPORT_SET KFCoreTargets
  FOLDER "thig")

if(BUILD_TESTS)
  add_executable(test_thig_temporal_graph
    tests/test_temporal_graph.cpp)
  target_include_directories(test_thig_temporal_graph PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_thig_temporal_graph PRIVATE
    KFCore::thig
    TurboUtils::TinyTest)
  target_compile_features(test_thig_temporal_graph PRIVATE cxx_std_17)
  add_test(NAME test_thig_temporal_graph COMMAND test_thig_temporal_graph)
endif()

cmake_install_headers(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERNS "*.hpp")
