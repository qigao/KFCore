include(CMakeDependentOption)

set(CMAKE_COLOR_DIAGNOSTICS ON)

option(KFCORE_BUILD_TESTS "Build KFCore C tests" ON)
option(KFCORE_BUILD_NAVIGATION_TOOLS "Build navigation helper functions" ON)
option(KFCORE_ENABLE_WARNINGS "Enable compiler warnings for KFCore targets" ON)
option(KFCORE_INSTALL "Generate KFCore install and CMake package rules" ON)
option(KFCORE_USE_SIMDE "Use SIMDe helpers in selected mini-BLAS kernels" OFF)

set(ENABLE_TESTS "${KFCORE_BUILD_TESTS}" CACHE BOOL "Compatibility alias for KFCORE_BUILD_TESTS" FORCE)
set(BUILD_TESTS "${KFCORE_BUILD_TESTS}" CACHE BOOL "Compatibility alias for KFCORE_BUILD_TESTS" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build example programs" FORCE)
