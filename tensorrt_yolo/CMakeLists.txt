add_library(kfcore_yolo_tracking SHARED
  src/engine_contract.cpp
  src/tracking.cpp)
add_library(KFCore::yolo_tracking ALIAS kfcore_yolo_tracking)

target_link_libraries(kfcore_yolo_tracking PRIVATE KFCore::trackers)
target_include_directories(kfcore_yolo_tracking PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(kfcore_yolo_tracking PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  EXPORT_NAME yolo_tracking
  FOLDER "tensorrt_yolo"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(BUILD_TESTS)
  add_executable(test_tensorrt_contract tests/test_tensorrt_contract.cpp)
  target_include_directories(test_tensorrt_contract PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_tensorrt_contract PRIVATE
    KFCore::yolo_tracking
    TurboUtils::TinyTest)
  set_target_properties(test_tensorrt_contract PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo/tests")
  add_test(NAME test_tensorrt_contract COMMAND test_tensorrt_contract)

  add_executable(test_tensorrt_detection_helpers
    tests/test_tensorrt_detection_helpers.cpp
    src/detector_helpers.cpp)
  target_include_directories(test_tensorrt_detection_helpers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_tensorrt_detection_helpers PRIVATE
    KFCore::yolo_tracking
    TurboUtils::TinyTest)
  set_target_properties(test_tensorrt_detection_helpers PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo/tests")
  add_test(NAME test_tensorrt_detection_helpers COMMAND test_tensorrt_detection_helpers)

  add_executable(test_yolo_tracking
    tests/test_yolo_tracking.cpp
    src/tracking.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../trackers/src/tracker.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../trackers/tests/tracker_test_alloc.c)
  target_compile_definitions(test_yolo_tracking PRIVATE KFCORE_TRACKERS_TEST_ALLOCATOR)
  target_include_directories(test_yolo_tracking PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../trackers/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../trackers/tests)
  target_link_libraries(test_yolo_tracking PRIVATE
    KFCore::kfcore
    miniblas
    TurboUtils::TinyTest)
  set_target_properties(test_yolo_tracking PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "tensorrt_yolo/tests")
  add_test(NAME test_yolo_tracking COMMAND test_yolo_tracking)
endif()

if(KFCORE_BUILD_TENSORRT_YOLO)
  add_library(tensorrt_yolo SHARED
    src/cuda_buffer.cpp
    src/detector_helpers.cpp
    src/detector.cpp
    src/engine_file.cpp
    src/engine.cpp
    src/letterbox.cu)
  add_library(KFCore::tensorrt_yolo ALIAS tensorrt_yolo)

  target_include_directories(tensorrt_yolo
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(tensorrt_yolo
    PUBLIC KFCore::yolo_tracking
    PRIVATE
      TensorRT::nvinfer
      TensorRT::nvinfer_plugin
      CUDA::cudart)
  set_target_properties(tensorrt_yolo PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    EXPORT_NAME tensorrt_yolo
    FOLDER "tensorrt_yolo"
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

  if(KFCORE_BUILD_TENSORRT_INTEGRATION_TESTS)
    add_executable(test_cuda_buffer
      tests/test_cuda_buffer.cpp
      src/cuda_buffer.cpp)
    target_include_directories(test_cuda_buffer PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(test_cuda_buffer PRIVATE
      KFCORE_YOLO_CUDA_BUFFER_TESTING)
    target_link_libraries(test_cuda_buffer PRIVATE
      KFCore::yolo_tracking
      CUDA::cudart
      TurboUtils::TinyTest)
    set_target_properties(test_cuda_buffer PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_yolo/tests")
    add_test(NAME test_cuda_buffer COMMAND test_cuda_buffer)

    add_executable(test_engine_file
      tests/test_engine_file.cpp
      src/engine_file.cpp)
    target_include_directories(test_engine_file PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_engine_file PRIVATE
      KFCore::yolo_tracking
      TurboUtils::TinyTest)
    set_target_properties(test_engine_file PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_yolo/tests")
    add_test(NAME test_engine_file COMMAND test_engine_file)

    add_executable(test_tensorrt_integration
      tests/test_tensorrt_integration.cpp)
    target_link_libraries(test_tensorrt_integration PRIVATE
      KFCore::tensorrt_yolo
      CUDA::cudart
      TurboUtils::TinyTest)
    set_target_properties(test_tensorrt_integration PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_yolo/tests")
    add_test(NAME test_tensorrt_integration COMMAND test_tensorrt_integration)

    add_executable(test_letterbox_cuda
      tests/test_letterbox_cuda.cu)
    target_include_directories(test_letterbox_cuda PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_letterbox_cuda PRIVATE
      KFCore::tensorrt_yolo
      CUDA::cudart
      TurboUtils::TinyTest)
    set_target_properties(test_letterbox_cuda PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      FOLDER "tensorrt_yolo/tests")
    add_test(NAME test_letterbox_cuda COMMAND test_letterbox_cuda)
  endif()

  install(TARGETS tensorrt_yolo
    EXPORT KFCoreTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

install(TARGETS kfcore_yolo_tracking
  EXPORT KFCoreTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
