add_library(kfcore_esn STATIC esn.c esn_reservoir.c esn_rls.c esn_sparse.c esn_group.c esn.h esn_sparse.h esn_group.h)

target_include_directories(
  kfcore_esn
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kfcore>)

target_compile_features(kfcore_esn PUBLIC c_std_11)
target_link_libraries(kfcore_esn PRIVATE KFCore::miniblas_internal)

target_compile_options(
  kfcore_esn
  PRIVATE $<$<C_COMPILER_ID:MSVC>:/W4>
          $<$<NOT:$<C_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wdouble-promotion>)

if(NOT WIN32)
  target_link_libraries(kfcore_esn PUBLIC m)
endif()

cmake_config_target(
  kfcore_esn
  ALIAS KFCore::esn
  EXPORT_NAME esn
  EXPORT_SET KFCoreTargets
  FOLDER "esn")

install(FILES esn.h esn_sparse.h esn_group.h DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kfcore")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
