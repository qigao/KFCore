add_library(kfcore_hand_interaction STATIC
  src/gesture_graph.cpp
  src/hand_interaction.cpp
  src/hand_identity.cpp
  src/hand_shape_descriptor.cpp
  src/primitive_extractor.cpp)
target_include_directories(kfcore_hand_interaction
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_hand_interaction PUBLIC cxx_std_17)
target_link_libraries(kfcore_hand_interaction
  PUBLIC
    KFCore::hand_model_core
    KFCore::thig)
set_target_properties(kfcore_hand_interaction PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/interaction/hand")
cmake_config_target(kfcore_hand_interaction
  ALIAS KFCore::hand_interaction
  EXPORT_NAME hand_interaction
  EXPORT_SET KFCoreTargets
  FOLDER "vision/interaction/hand")

if(BUILD_EXAMPLES)
 
  find_package(OpenCVLite REQUIRED COMPONENTS core imgproc highgui)

  add_executable(hand_interaction_demo
    examples/hand_interaction_demo.cpp
    examples/hand_interaction_demo_cli.cpp
    examples/hand_interaction_demo_capture.cpp
    examples/hand_interaction_demo_face.cpp
    examples/hand_interaction_demo_frame.cpp
    examples/hand_interaction_demo_ui.cpp)
  target_include_directories(hand_interaction_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples)
  target_link_libraries(hand_interaction_demo PRIVATE
    KFCore::hand_interaction
    OpenCVLite::core
    OpenCVLite::imgproc
    OpenCVLite::highgui
    Salts::Capture
    Salts::Core)
  if(KFCORE_ENABLE_ONNX_CPU)
    target_link_libraries(hand_interaction_demo PRIVATE
      KFCore::hand_models_cpu
      KFCore::face_models_cpu)
  endif()
  if(KFCORE_ENABLE_CUDA)
    target_link_libraries(hand_interaction_demo PRIVATE
      KFCore::hand_models_cuda
      KFCore::face_models_cuda)
  endif()
  target_compile_definitions(hand_interaction_demo PRIVATE
    KFCORE_DEMO_ENABLE_CUDA=$<BOOL:${KFCORE_ENABLE_CUDA}>
    KFCORE_DEMO_ENABLE_ONNX_CPU=$<BOOL:${KFCORE_ENABLE_ONNX_CPU}>)
  set_target_properties(hand_interaction_demo PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  cmake_config_target(hand_interaction_demo
    NO_INSTALL FOLDER "vision/interaction/hand/examples")
endif()

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERNS "*.hpp")
