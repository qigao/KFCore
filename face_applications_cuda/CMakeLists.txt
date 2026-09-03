add_library(kfcore_face_applications_cuda SHARED
  src/composer.cpp
  src/error.cpp
  src/geometry.cpp)
target_sources(kfcore_face_applications_cuda PRIVATE src/tensorrt.cpp)
target_include_directories(kfcore_face_applications_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(kfcore_face_applications_cuda PUBLIC cxx_std_17)
target_link_libraries(kfcore_face_applications_cuda
  PUBLIC
    KFCore::image_processor_cuda
    KFCore::face_models_cuda
    KFCore::yolo_tensorrt)
set_target_properties(kfcore_face_applications_cuda PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "face_applications_cuda"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
cmake_config_target(kfcore_face_applications_cuda
  ALIAS KFCore::face_applications_cuda
  EXPORT_NAME face_applications_cuda
  EXPORT_SET KFCoreTargets
  FOLDER "face_applications_cuda")

if(BUILD_EXAMPLES)
  find_package(OpenCVLite REQUIRED COMPONENTS core imgproc imgcodecs highgui)
  add_executable(face_swap_image
    examples/face_swap_cli.cpp
    examples/face_swap_image.cpp)
  target_include_directories(face_swap_image PRIVATE examples)
  target_link_libraries(face_swap_image PRIVATE
    KFCore::face_applications_cuda
    OpenCVLite::core
    OpenCVLite::imgcodecs
    Salts::Core)
  set_target_properties(face_swap_image PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  cmake_config_target(face_swap_image
    NO_INSTALL FOLDER "face_applications_cuda/examples")

  add_executable(face_swap_demo
    examples/face_swap_cli.cpp
    examples/face_swap_demo.cpp
    examples/face_swap_demo_metrics.cpp
    examples/face_swap_demo_ui.cpp)
  target_include_directories(face_swap_demo PRIVATE examples)
  target_link_libraries(face_swap_demo PRIVATE
    KFCore::face_applications_cuda
    OpenCVLite::core
    OpenCVLite::imgproc
    OpenCVLite::imgcodecs
    OpenCVLite::highgui
    Salts::Core)
  set_target_properties(face_swap_demo PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  cmake_config_target(face_swap_demo
    NO_INSTALL FOLDER "face_applications_cuda/examples")
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")
