cmake_add_test(
  NAME kfcore_esn_tests
  SOURCES test_esn.c test_esn_rls.c test_esn_sparse.c test_esn_sparse_errors.c test_esn_group.c test_esn_deep.c
  LIBS KFCore::esn Salts::TinyTest
  FOLDER "esn/tests"
  TARGET_PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF)
