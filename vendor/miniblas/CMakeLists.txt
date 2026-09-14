add_library(miniblas STATIC miniblas.c linalg.c miniblas.h linalg.h)

target_include_directories(miniblas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

find_path(SIMDE_INCLUDE_DIRS "simde/simde-common.h" REQUIRED)
target_include_directories(miniblas PRIVATE ${SIMDE_INCLUDE_DIRS})

set_target_properties(
  miniblas
  PROPERTIES C_STANDARD 11
             C_STANDARD_REQUIRED ON
             FOLDER vendor)
cmake_config_target(miniblas
  ALIAS KFCore::miniblas_internal
  EXPORT_NAME miniblas_internal
  EXPORT_SET KFCoreTargets
  FOLDER "vendor")
