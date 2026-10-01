add_library(kfcore_relation_core STATIC
  src/decode.cpp
  src/vocabulary_score.cpp
  src/predicate_text_encoder.cpp
  src/predicate_routing_gate.cpp
  src/relate_anything.cpp
  src/open_vocabulary_relation.cpp)
target_include_directories(kfcore_relation_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(kfcore_relation_core PUBLIC cxx_std_17)
target_link_libraries(kfcore_relation_core
  PUBLIC
    KFCore::runtime_core
    KFCore::image_processor_core
  PRIVATE
    KFCore::image_processor_cpu)
set_target_properties(kfcore_relation_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  FOLDER "vision/core/relation")
cmake_config_target(kfcore_relation_core
  ALIAS KFCore::relation_core
  EXPORT_NAME relation_core
  EXPORT_SET KFCoreTargets
  FOLDER "vision/core/relation")

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()

cmake_install_headers(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PATTERNS "*.hpp")

if(KFCORE_ENABLE_HF_TOKENIZER_PROVIDER)
  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(_kfcore_tokenizer_android_abis
      arm64-v8a
      armeabi-v7a
      x86_64
      x86)
    if(NOT ANDROID_ABI IN_LIST _kfcore_tokenizer_android_abis)
      message(FATAL_ERROR
        "KFCore HF tokenizer provider supports Android ABI "
        "arm64-v8a/armeabi-v7a/x86_64/x86; got '${ANDROID_ABI}'")
    endif()
    unset(_kfcore_tokenizer_android_abis)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    string(TOUPPER "${CMAKE_SYSTEM_PROCESSOR}" _kfcore_tokenizer_processor)
    if(NOT _kfcore_tokenizer_processor STREQUAL "AMD64" AND
       NOT _kfcore_tokenizer_processor STREQUAL "X86_64" AND
       NOT _kfcore_tokenizer_processor STREQUAL "ARM64" AND
       NOT _kfcore_tokenizer_processor STREQUAL "AARCH64")
      message(FATAL_ERROR
        "KFCore HF tokenizer provider supports Windows x86_64/ARM64 only; "
        "got '${CMAKE_SYSTEM_PROCESSOR}'")
    endif()
    unset(_kfcore_tokenizer_processor)
  endif()

  if(KFCORE_TOKENIZERS_CPP_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR
      "KFCORE_ENABLE_HF_TOKENIZER_PROVIDER requires KFCORE_TOKENIZERS_CPP_SOURCE_DIR")
  endif()
  if(NOT EXISTS "${KFCORE_TOKENIZERS_CPP_SOURCE_DIR}/CMakeLists.txt" OR
     NOT EXISTS "${KFCORE_TOKENIZERS_CPP_SOURCE_DIR}/include/tokenizers_c.h")
    message(FATAL_ERROR
      "KFCORE_TOKENIZERS_CPP_SOURCE_DIR must point to a tokenizers-cpp checkout")
  endif()

  if(NOT TARGET tokenizers_c)
    set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
    add_subdirectory(
      "${KFCORE_TOKENIZERS_CPP_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/tokenizers-cpp"
      EXCLUDE_FROM_ALL)
  endif()

  add_library(kfcore_relation_tokenizer_hf STATIC
    src/predicate_tokenizer_hf.cpp)
  target_include_directories(kfcore_relation_tokenizer_hf
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
      "${KFCORE_TOKENIZERS_CPP_SOURCE_DIR}/include")
  target_compile_features(kfcore_relation_tokenizer_hf PUBLIC cxx_std_17)
  target_link_libraries(kfcore_relation_tokenizer_hf
    PUBLIC
      KFCore::relation_core
    PRIVATE
      tokenizers_c)
  set_target_properties(kfcore_relation_tokenizer_hf PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    FOLDER "vision/core/relation")
  add_library(KFCore::relation_tokenizer_hf ALIAS
    kfcore_relation_tokenizer_hf)
endif()
