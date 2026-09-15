if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
  set(PopSift_HAVE_SHFL_DOWN_SYNC 0)
else()
  set(PopSift_HAVE_SHFL_DOWN_SYNC 1)
endif()
set(DISABLE_GRID_FILTER 0)

set(_popsift_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(KFCORE_SOURCE_REVISION "" CACHE STRING
  "Full KFCore commit used in installed source provenance when Git metadata is unavailable")
set(_kfcore_source_revision "${KFCORE_SOURCE_REVISION}")
execute_process(
  COMMAND git -C "${PROJECT_SOURCE_DIR}" rev-parse --verify HEAD
  RESULT_VARIABLE _kfcore_revision_result
  OUTPUT_VARIABLE _kfcore_git_revision
  ERROR_VARIABLE _kfcore_revision_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
set(_kfcore_matching_git_checkout FALSE)
if(_kfcore_revision_result EQUAL 0 AND NOT _kfcore_git_revision STREQUAL "")
  execute_process(
    COMMAND git -C "${PROJECT_SOURCE_DIR}" rev-parse --show-toplevel
    RESULT_VARIABLE _kfcore_toplevel_result
    OUTPUT_VARIABLE _kfcore_git_toplevel
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_kfcore_toplevel_result EQUAL 0)
    file(REAL_PATH "${PROJECT_SOURCE_DIR}" _kfcore_source_real_path)
    file(REAL_PATH "${_kfcore_git_toplevel}" _kfcore_git_real_path)
    if(_kfcore_source_real_path STREQUAL _kfcore_git_real_path)
      set(_kfcore_matching_git_checkout TRUE)
    endif()
  endif()
endif()
if(_kfcore_matching_git_checkout)
  string(LENGTH "${_kfcore_git_revision}" _kfcore_git_revision_length)
  if(NOT _kfcore_git_revision_length EQUAL 40 OR
     NOT _kfcore_git_revision MATCHES "^[0-9A-Fa-f]+$")
    message(FATAL_ERROR "Git returned a non-full KFCore commit: ${_kfcore_git_revision}")
  endif()
  if(NOT _kfcore_source_revision STREQUAL "" AND
     NOT _kfcore_source_revision STREQUAL _kfcore_git_revision)
    message(FATAL_ERROR
      "KFCORE_SOURCE_REVISION (${_kfcore_source_revision}) does not match "
      "the current Git HEAD (${_kfcore_git_revision})")
  endif()
  set(_kfcore_source_revision "${_kfcore_git_revision}")

  execute_process(
    COMMAND git -C "${PROJECT_SOURCE_DIR}" rev-parse --git-path HEAD
    OUTPUT_VARIABLE _kfcore_git_head_path
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  file(REAL_PATH "${_kfcore_git_head_path}" _kfcore_git_head_path
    BASE_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_kfcore_git_head_path}")
  execute_process(
    COMMAND git -C "${PROJECT_SOURCE_DIR}" symbolic-ref -q HEAD
    RESULT_VARIABLE _kfcore_symbolic_ref_result
    OUTPUT_VARIABLE _kfcore_symbolic_ref
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_kfcore_symbolic_ref_result EQUAL 0 AND NOT _kfcore_symbolic_ref STREQUAL "")
    execute_process(
      COMMAND git -C "${PROJECT_SOURCE_DIR}" rev-parse --git-path "${_kfcore_symbolic_ref}"
      OUTPUT_VARIABLE _kfcore_git_ref_path
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    file(REAL_PATH "${_kfcore_git_ref_path}" _kfcore_git_ref_path
      BASE_DIRECTORY "${PROJECT_SOURCE_DIR}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${_kfcore_git_ref_path}")
  endif()
else()
  string(LENGTH "${_kfcore_source_revision}" _kfcore_revision_length)
  if(NOT _kfcore_revision_length EQUAL 40 OR
     NOT _kfcore_source_revision MATCHES "^[0-9A-Fa-f]+$")
    message(FATAL_ERROR
      "CUDA SIFT requires a full 40-hex KFCore commit for MPL provenance. "
      "Set KFCORE_SOURCE_REVISION when matching KFCore Git metadata is unavailable. Git error: "
      "${_kfcore_revision_error}")
  endif()
endif()
set(KFCORE_POPSIFT_SOURCE_REVISION "${_kfcore_source_revision}" PARENT_SCOPE)
set(KFCORE_POPSIFT_SOURCE_URL
  "https://github.com/qigao/KFCore/tree/${_kfcore_source_revision}/sift/vendor/popsift")
set(KFCORE_SOURCE_REVISION_RESOLVED "${_kfcore_source_revision}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/UPSTREAM.md"
  "${_popsift_generated_dir}/UPSTREAM.md"
  @ONLY)
set(KFCORE_CONFIGURED_SOURCE_REVISION "${_kfcore_source_revision}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_source_revision.cmake.in"
  "${_popsift_generated_dir}/verify_source_revision.cmake"
  @ONLY)
set(KFCORE_POPSIFT_REVISION_VERIFY_SCRIPT
  "${_popsift_generated_dir}/verify_source_revision.cmake" PARENT_SCOPE)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/sift_config.h.in"
  "${_popsift_generated_dir}/popsift/sift_config.h"
  @ONLY)

add_library(kfcore_popsift_internal STATIC
  src/popsift/popsift.cu
  src/popsift/features.cu
  src/popsift/sift_constants.cu
  src/popsift/sift_conf.cu
  src/popsift/gauss_filter.cu
  src/popsift/s_image.cu
  src/popsift/sift_pyramid.cu
  src/popsift/sift_octave.cu
  src/popsift/s_pyramid_build.cu
  src/popsift/s_pyramid_build_aa.cu
  src/popsift/s_pyramid_build_ai.cu
  src/popsift/s_pyramid_build_ra.cu
  src/popsift/s_pyramid_fixed.cu
  src/popsift/sift_extremum.cu
  src/popsift/s_extrema.cu
  src/popsift/s_orientation.cu
  src/popsift/s_filtergrid.cu
  src/popsift/sift_desc.cu
  src/popsift/s_desc_loop.cu
  src/popsift/s_desc_iloop.cu
  src/popsift/s_desc_grid.cu
  src/popsift/s_desc_igrid.cu
  src/popsift/s_desc_notile.cu
  src/popsift/s_desc_vlfeat.cu
  src/popsift/common/assist.cu
  src/popsift/common/plane_2d.cu
  src/popsift/common/write_plane_2d.cu
  src/popsift/common/debug_macros.cu
  src/popsift/common/device_prop.cu)

target_include_directories(kfcore_popsift_internal PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${_popsift_generated_dir}>
  $<BUILD_INTERFACE:${_popsift_generated_dir}/popsift>)
target_link_libraries(kfcore_popsift_internal
  PRIVATE CUDA::cudart_static Threads::Threads)
target_compile_features(kfcore_popsift_internal PUBLIC cxx_std_17)
set_target_properties(kfcore_popsift_internal PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
  FOLDER "sift/internal")
cmake_config_target(kfcore_popsift_internal
  ALIAS KFCore::popsift_internal
  EXPORT_NAME popsift_internal
  EXPORT_SET KFCoreTargets
  FOLDER "sift/internal")
if(MSVC)
  target_compile_options(kfcore_popsift_internal PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
if(BUILD_TESTS)
  target_compile_definitions(kfcore_popsift_internal PRIVATE
    KFCORE_POPSIFT_TESTING=1)
endif()
