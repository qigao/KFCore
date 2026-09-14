add_library(kfcore_backend_tensorrt SHARED
  src/plugin.cpp)
target_compile_features(kfcore_backend_tensorrt PRIVATE cxx_std_17)
target_link_libraries(kfcore_backend_tensorrt PRIVATE
  KFCore::runtime_abi
  KFCore::runtime_tensorrt
  TensorRT::nvinfer
  CUDA::cudart_static)
target_include_directories(kfcore_backend_tensorrt PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/abi/include)
set_target_properties(kfcore_backend_tensorrt PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  OUTPUT_NAME "kfcore_backend_tensorrt"
  FOLDER "backends/tensorrt/plugin")
if(UNIX AND NOT APPLE)
  set_target_properties(kfcore_backend_tensorrt PROPERTIES
    INSTALL_RPATH "$ORIGIN")
endif()
cmake_config_target(kfcore_backend_tensorrt
  PLUGIN
  FOLDER "backends/tensorrt/plugin")

if(WIN32)
  file(REAL_PATH "$ENV{TENSORRT_ROOT}" _kfcore_tensorrt_root)
  cmake_path(SET _kfcore_tensorrt_root NORMALIZE "${_kfcore_tensorrt_root}")
  set(_kfcore_tensorrt_runtime_paths
    "${_kfcore_tensorrt_root}/bin"
    "${_kfcore_tensorrt_root}/lib"
    "${_kfcore_tensorrt_root}/lib/x64")

  function(_kfcore_find_tensorrt_runtime_dll library_stem output_variable)
    unset(_kfcore_tensorrt_runtime_dll CACHE)
    find_file(_kfcore_tensorrt_runtime_dll
      NAMES
        "${library_stem}_${TensorRT_VERSION_MAJOR}.dll"
        "${library_stem}.dll"
      PATHS ${_kfcore_tensorrt_runtime_paths}
      NO_DEFAULT_PATH)
    if(NOT _kfcore_tensorrt_runtime_dll)
      message(FATAL_ERROR
        "TENSORRT_ROOT does not contain ${library_stem} runtime DLL for TensorRT "
        "${TensorRT_VERSION_MAJOR}: ${_kfcore_tensorrt_root}")
    endif()
    file(REAL_PATH "${_kfcore_tensorrt_runtime_dll}" _kfcore_runtime_dll_real)
    cmake_path(SET _kfcore_runtime_dll_real NORMALIZE "${_kfcore_runtime_dll_real}")
    cmake_path(IS_PREFIX _kfcore_tensorrt_root "${_kfcore_runtime_dll_real}"
               NORMALIZE _kfcore_runtime_dll_contained)
    if(NOT _kfcore_runtime_dll_contained)
      message(FATAL_ERROR
        "TensorRT runtime DLL resolved outside TENSORRT_ROOT: ${_kfcore_runtime_dll_real}")
    endif()
    set(${output_variable} "${_kfcore_runtime_dll_real}" PARENT_SCOPE)
    unset(_kfcore_tensorrt_runtime_dll CACHE)
  endfunction()

  _kfcore_find_tensorrt_runtime_dll(nvinfer _kfcore_nvinfer_runtime_dll)
  _kfcore_find_tensorrt_runtime_dll(nvinfer_plugin _kfcore_nvinfer_plugin_runtime_dll)

  add_custom_command(TARGET kfcore_backend_tensorrt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_kfcore_nvinfer_runtime_dll}"
            "$<TARGET_FILE_DIR:kfcore_backend_tensorrt>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_kfcore_nvinfer_plugin_runtime_dll}"
            "$<TARGET_FILE_DIR:kfcore_backend_tensorrt>"
    VERBATIM)

  install(FILES
    "${_kfcore_nvinfer_runtime_dll}"
    "${_kfcore_nvinfer_plugin_runtime_dll}"
    DESTINATION "${KFCORE_INSTALL_PLUGINDIR}")
endif()
