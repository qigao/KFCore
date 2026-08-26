if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
  set(PopSift_HAVE_SHFL_DOWN_SYNC 0)
else()
  set(PopSift_HAVE_SHFL_DOWN_SYNC 1)
endif()
set(DISABLE_GRID_FILTER 0)

set(_popsift_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
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
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${_popsift_generated_dir}"
  "${_popsift_generated_dir}/popsift")
target_link_libraries(kfcore_popsift_internal
  PUBLIC CUDA::cudart Threads::Threads)
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
if(MSVC)
  target_compile_options(kfcore_popsift_internal PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
