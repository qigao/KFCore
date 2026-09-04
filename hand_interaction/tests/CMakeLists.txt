set(_hand_test_folder "hand_interaction/tests")

cmake_add_test(
  NAME test_hand_primitive_extractor
  SOURCES test_primitive_extractor.cpp
  LIBS KFCore::hand_interaction Salts::TinyTest
  INCLUDES ../include
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_test_folder}")
cmake_add_test(
  NAME test_hand_interaction
  SOURCES test_hand_interaction.cpp
  LIBS KFCore::hand_interaction Salts::TinyTest
  INCLUDES ../include
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_test_folder}")

cmake_add_test(
  NAME test_hand_interaction_demo_cli
  SOURCES test_hand_interaction_demo_cli.cpp ../examples/hand_interaction_demo_cli.cpp
  LIBS
    KFCore::hand_models_cpu
    KFCore::hand_models_cuda
    Salts::Capture
    Salts::Core
    Salts::TinyTest
  INCLUDES ../examples
  COMPILE_FEATURES cxx_std_17
  FOLDER "${_hand_test_folder}")

if(BUILD_EXAMPLES)
  cmake_add_test(
    NAME test_hand_interaction_demo_capture
    SOURCES
      test_hand_interaction_demo_capture.cpp
      ../examples/hand_interaction_demo_capture.cpp
    LIBS
      KFCore::image_processor_core
      Salts::Capture
      Salts::Core
      Salts::TinyTest
    INCLUDES ../examples
    COMPILE_FEATURES cxx_std_17
    FOLDER "${_hand_test_folder}")
  cmake_add_test(
    NAME test_hand_interaction_demo_ui
    SOURCES
      test_hand_interaction_demo_ui.cpp
      ../examples/hand_interaction_demo_capture.cpp
      ../examples/hand_interaction_demo_frame.cpp
      ../examples/hand_interaction_demo_ui.cpp
    LIBS
      KFCore::hand_interaction
      KFCore::face_model_core
      OpenCVLite::core
      OpenCVLite::imgproc
      Salts::Capture
      Salts::Core
      Salts::TinyTest
    INCLUDES ../examples
    COMPILE_FEATURES cxx_std_17
    FOLDER "${_hand_test_folder}")
endif()
