# CMakeLists.txt for the KFCore Kalman filter unit tests.
#
# The test executable links against the `kfcore` library (via the KFCore::kfcore
# alias) and the Salts::TinyTest framework that provides the tinytest.h
# header and test runner.

cmake_add_test(
    NAME kfcore_kalman_tests
    SOURCES test.c
    LIBS KFCore::kfcore Salts::TinyTest miniblas apriltag
    FOLDER "kalman/tests"
    TARGET_PROPERTIES
      C_STANDARD 11
      C_STANDARD_REQUIRED ON
      C_EXTENSIONS OFF)
