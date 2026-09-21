cmake_add_test(
  NAME kfcore_esn_tests
  SOURCES test_esn.c test_esn_rls.c test_esn_sparse.c test_esn_sparse_errors.c test_esn_group.c test_esn_deep.c test_esn_group_deep.c
  LIBS KFCore::esn Salts::TinyTest
  FOLDER "esn/tests"
  TARGET_PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF)

add_executable(kfcore_esn_sequence_tests test_esn_sequence.c)
target_link_libraries(kfcore_esn_sequence_tests PRIVATE KFCore::esn Salts::TinyTest)
add_test(NAME kfcore_esn_sequence_tests COMMAND kfcore_esn_sequence_tests)

add_executable(kfcore_esn_quality EXCLUDE_FROM_ALL test_esn_sequence.c)
target_compile_definitions(kfcore_esn_quality PRIVATE KFCORE_ESN_QUALITY=1)
target_link_libraries(kfcore_esn_quality PRIVATE KFCore::esn Salts::TinyTest)
